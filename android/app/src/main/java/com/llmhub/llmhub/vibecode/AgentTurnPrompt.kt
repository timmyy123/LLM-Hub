package com.llmhub.llmhub.vibecode

/**
 * The request Codex and Claude Code modes send for one turn: the open file and the user's words, held to the same bar
 * as Vibe Code's normal mode. Without it, agent prompts are all protocol (read, edit small, verify) and small models
 * create minimal stubs.
 */
internal object AgentTurnPrompt {
    /**
     * [shownPath]: how the file is named to the agent (relative for Codex, absolute for Claude Code).
     * [target]: normal mode's rule for the file's language, e.g. "Build a single self-contained HTML file…".
     * [writeTool]/[editTool]: the agent's own names for writing a whole file and replacing a fragment.
     */
    fun build(
        prompt: String, shownPath: String?, editorName: String?, editorCode: String, contextWindow: Int,
        target: String?, writeTool: String, editTool: String
    ): String = buildString {
        if (!editorName.isNullOrBlank()) {
            append("Active editor file: ").append(shownPath ?: editorName).append("\n")
            target?.let { append("Target: ").append(it).append("\n") }
            // Small models edit from memory instead of reading first, dropping everything they forgot.
            // A file that fits comfortably in the context is given to them as it is now.
            if (editorCode.isEmpty()) {
                append(editorName).append(" is currently empty. Create its complete contents with ").append(writeTool).append(": ")
                append("a complete, working, polished implementation of the request with real content, structure and styling, ")
                append("not a minimal stub or placeholder.\n")
            } else if (editorCode.length <= contextWindow * 3 / 4) {
                append("Current contents of ").append(editorName).append(":\n```\n").append(editorCode).append("\n```\n")
                append("Keep everything the request does not change. Use ").append(editTool).append(" with an exact fragment from these ")
                append("contents, or ").append(writeTool).append(" with the complete updated file. The result must stay a complete, working file.\n")
            } else {
                append("Read the current file from disk using tools. Its contents may have changed since earlier turns.\n")
            }
            append("\nUser request:\n")
        }
        append(prompt)
    }
}
