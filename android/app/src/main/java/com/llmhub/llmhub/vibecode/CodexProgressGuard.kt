package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject

/** Prevent a successful inspection from being executed forever while the model repeats its plan. */
internal object CodexProgressGuard {
    /** End a redundant verification only when a literal write and its exact read-back succeeded. */
    fun verifiedDuplicateRead(output: JSONArray, request: JSONObject): String? {
        val proposed = (0 until output.length()).map { output.getJSONObject(it) }
            .filter { it.optString("type") in setOf("function_call", "custom_tool_call") }
        if (proposed.size != 1) return null
        val nextKey = inspectionKey(proposed.single())?.takeIf { it.startsWith("cat ") } ?: return null
        val input = request.optJSONArray("input") ?: return null
        val calls = mutableMapOf<String, JSONObject>()
        var written: Pair<String, String>? = null
        var verifiedKey: String? = null
        for (i in 0 until input.length()) {
            val item = input.optJSONObject(i) ?: continue
            if (item.optString("role") == "user") { written = null; verifiedKey = null; calls.clear() }
            when (item.optString("type")) {
                "function_call" -> calls[item.optString("call_id")] = item
                "function_call_output" -> {
                    val call = calls[item.optString("call_id")] ?: continue
                    val result = item.optString("output")
                    val key = inspectionKey(call)
                    if (key == null) {
                        verifiedKey = null
                        written = if (completeInspection(result)) command(call)?.let(CodexFileEdits::writtenFile) else null
                    } else if (written != null && key == "cat ${written.first}") {
                        val content = result.substringAfter("Output:\n", "")
                        verifiedKey = if (completeInspection(result) && result.contains("Output:\n") &&
                            (content == written.second || content == written.second + "\n")) key else null
                    }
                }
            }
        }
        return written?.first?.takeIf { nextKey == verifiedKey }
    }
    private fun completeInspection(output: String): Boolean = output.contains("Process exited with code 0") &&
        !Regex("truncated output|tokens truncated|output truncated", RegexOption.IGNORE_CASE).containsMatchIn(output)
    fun latestResult(request: JSONObject): Pair<JSONObject, String>? {
        val input = request.optJSONArray("input") ?: return null
        val calls = mutableMapOf<String, JSONObject>()
        var latest: Pair<JSONObject, String>? = null
        for (i in 0 until input.length()) {
            val item = input.optJSONObject(i) ?: continue
            if (item.optString("role") == "user") latest = null
            when (item.optString("type")) {
                "function_call", "custom_tool_call" -> calls[item.optString("call_id")] = item
                "function_call_output", "custom_tool_call_output" -> {
                    val call = calls[item.optString("call_id")] ?: continue
                    latest = call to item.optString("output")
                }
            }
        }
        return latest
    }

    internal fun command(call: JSONObject): String? {
        val args = runCatching { JSONObject(call.optString("arguments")) }.getOrNull() ?: return null
        return args.optString("cmd").ifBlank { args.optString("command") }.trim().takeIf { it.isNotEmpty() }
    }

    internal fun inspectionKey(call: JSONObject): String? {
        val cmd = command(call) ?: return null
        if (!Regex("^(?:cat|head|tail|ls|pwd|wc|grep|rg|diff)\\b").containsMatchIn(cmd) || Regex("[;|&><]").containsMatchIn(cmd)) return null
        val executable = cmd.substringBefore(' ')
        val args = cmd.substringAfter(' ', "").trim().removeSurrounding("\"").removeSurrounding("'")
        return "$executable $args".trim()
    }

    /** Preserve the latest file snapshot; older successful reads of that same file are stale. */
    fun supersededResults(request: JSONObject): Set<String> {
        val input = request.optJSONArray("input") ?: return emptySet()
        val calls = mutableMapOf<String, JSONObject>()
        val latest = mutableMapOf<String, String>()
        val superseded = mutableSetOf<String>()
        for (i in 0 until input.length()) {
            val item = input.optJSONObject(i) ?: continue
            when (item.optString("type")) {
                "function_call" -> calls[item.optString("call_id")] = item
                "function_call_output" -> {
                    val id = item.optString("call_id")
                    val call = calls[id] ?: continue
                    val key = inspectionKey(call)
                    if (key == null) {
                        if (item.optString("output").contains("Process exited with code 0") &&
                            !Regex("^(cat|head|tail|ls|pwd|wc)\\b").containsMatchIn(command(call).orEmpty())) {
                            superseded.addAll(latest.values)
                            latest.clear()
                        }
                        continue
                    }
                    if (completeInspection(item.optString("output"))) {
                        latest.put(key, id)?.let(superseded::add)
                    }
                }
            }
        }
        return superseded
    }

    fun validate(output: JSONArray, request: JSONObject) {
        val input = request.optJSONArray("input") ?: return
        val calls = mutableMapOf<String, JSONObject>()
        val inspected = mutableSetOf<String>()
        val failedCommands = mutableSetOf<String>()
        var editingRequest = false
        var verifiedAction = false
        var actionNeedsVerification = false
        var lastAction: JSONObject? = null
        var actionSucceeded = false
        for (i in 0 until input.length()) {
            val item = input.optJSONObject(i) ?: continue
            if (item.optString("role") == "user") {
                inspected.clear(); failedCommands.clear(); lastAction = null; actionSucceeded = false
                verifiedAction = false
                actionNeedsVerification = false
                val content = item.optJSONArray("content")
                val userText = if (content == null) item.optString("content") else
                    (0 until content.length()).joinToString("\n") { content.optJSONObject(it)?.optString("text").orEmpty() }
                editingRequest = Regex("^\\s*(?:(?:please|can you|could you|i (?:ask|want|need) you to)\\s+)*(fix|edit|update|change|add|remove|replace|use|make|create|implement|convert|improve|refactor)\\b", RegexOption.IGNORE_CASE)
                    .containsMatchIn(userText.substringAfter("\nUser request:\n", userText).substringBefore("\n\n"))
            }
            when (item.optString("type")) {
                "function_call", "custom_tool_call" -> {
                    calls[item.optString("call_id")] = item
                    if (inspectionKey(item) == null) {
                        lastAction = item; actionSucceeded = false
                    }
                }
                "function_call_output", "custom_tool_call_output" -> {
                    val call = calls[item.optString("call_id")] ?: continue
                    if (!item.optString("output").contains("Process exited with code 0")) {
                        if (inspectionKey(call) == null && item.optString("output").contains("Process exited with code"))
                            command(call)?.let(failedCommands::add)
                        continue
                    }
                    if (inspectionKey(call) == null) { inspected.clear(); failedCommands.clear() }
                    val cmd = command(call).orEmpty()
                    val checkCommand = Regex("^(?:(?:npm|pnpm|yarn) (?:test|run (?:test|build|lint))|(?:\\./)?gradlew\\s|node --check\\s|python(?:3)? -m (?:pytest|unittest)\\s)").containsMatchIn(cmd)
                    if (checkCommand) verifiedAction = true
                    else if (inspectionKey(call) == null && !Regex("^(cat|head|tail|ls|pwd|wc)\\b").containsMatchIn(cmd)) {
                        actionNeedsVerification = true
                        verifiedAction = false
                    } else if (completeInspection(item.optString("output")) &&
                        (actionNeedsVerification || Regex("^(grep|rg|diff)\\b").containsMatchIn(cmd))) verifiedAction = true
                    if (completeInspection(item.optString("output"))) inspectionKey(call)?.let(inspected::add)
                    if (item.optString("call_id") == lastAction?.optString("call_id")) actionSucceeded = true
                }
            }
        }
        val finalReply = (0 until output.length()).none {
            output.getJSONObject(it).optString("type") in setOf("function_call", "custom_tool_call")
        }
        if (finalReply && editingRequest && !verifiedAction) throw IllegalArgumentException(
            "Unverified completion: no successful verification establishes this request is complete. " +
                "If you edited, check the resulting file or run relevant tests. If no edit was needed, run a targeted check (such as grep for the required import) that proves the request is already satisfied.")
        for (i in 0 until output.length()) {
            val next = output.getJSONObject(i)
            if (next.optString("type") != "function_call") continue
            val key = inspectionKey(next)
            if (key != null && key in inspected) {
                throw IllegalArgumentException("Repeated inspection: ${command(next)} already succeeded since the last change. " +
                    "Use the latest returned contents. Perform a necessary edit, or finish if the completed commands and verification satisfy the request. Do not read the same file again.")
            }
            val nextCommand = command(next) ?: continue
            if (nextCommand in failedCommands) throw IllegalArgumentException(
                "Repeated failed command: this command already failed and changed nothing. Do not repeat it or claim success. " +
                    "Use a smaller exact fragment from the file, or write_file with complete corrected contents.")
            if (key == null && actionSucceeded && inspected.isNotEmpty() && lastAction?.let(::command) == nextCommand) {
                throw IllegalArgumentException("Repeated edit: this exact command already succeeded and a subsequent inspection succeeded. " +
                    "Use the latest file contents, not older snapshots. If the requested fix is verified, return a final reply with tool_calls: []; otherwise make a different necessary change.")
            }
        }
    }
}
