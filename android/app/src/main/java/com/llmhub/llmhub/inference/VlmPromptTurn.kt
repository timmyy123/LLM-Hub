package com.llmhub.llmhub.inference

/** Role-labelled chat turn shared by the GGUF text and vision prompt builders. */
internal data class VlmPromptTurn(val role: String, val text: String)

/** Preserve the same transcript parsing used by the former GenieX service. */
internal fun parseVlmPromptTurns(prompt: String): List<VlmPromptTurn> {
    val cleaned = prompt.trim().let {
        if (it.endsWith("assistant:")) it.removeSuffix("assistant:").trimEnd() else it
    }
    if (cleaned.isBlank()) return emptyList()

    val roleMarker = Regex("(?m)^(system|user|assistant):[ \t]*", RegexOption.IGNORE_CASE)
    val matches = roleMarker.findAll(cleaned).toList()
    if (matches.isEmpty()) return listOf(VlmPromptTurn("user", cleaned))

    val turns = mutableListOf<VlmPromptTurn>()
    val prefix = cleaned.substring(0, matches.first().range.first).trim()
    if (prefix.isNotEmpty()) turns += VlmPromptTurn("system", prefix)

    matches.forEachIndexed { index, match ->
        val contentStart = match.range.last + 1
        val contentEnd = matches.getOrNull(index + 1)?.range?.first ?: cleaned.length
        val content = cleaned.substring(contentStart, contentEnd).trim()
        if (content.isNotEmpty()) {
            turns += VlmPromptTurn(match.groupValues[1].lowercase(), content)
        }
    }
    return turns
}
