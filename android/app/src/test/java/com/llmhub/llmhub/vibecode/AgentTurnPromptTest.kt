package com.llmhub.llmhub.vibecode

import org.junit.Assert.*
import org.junit.Test

class AgentTurnPromptTest {
    private val web = "Web App (HTML/CSS/JS). Build a single self-contained HTML file with embedded CSS and JavaScript."

    @Test fun aNewFileAsksForACompleteImplementationToNormalModesRule() {
        val prompt = AgentTurnPrompt.build("make a dog site", "/storage/emulated/0/Code/dog.html", "dog.html", "", 16384,
            web, writeTool = "Write", editTool = "Edit")
        assertTrue(prompt.contains("Active editor file: /storage/emulated/0/Code/dog.html"))
        assertTrue(prompt.contains("Target: $web"))
        assertTrue(prompt.contains("dog.html is currently empty. Create its complete contents with Write"))
        assertTrue(prompt.contains("not a minimal stub"))
        assertTrue(prompt.endsWith("\nUser request:\nmake a dog site"))
        // The Codex guard still recognises the empty file.
        assertTrue(Regex("(?m)^(\\S+) is currently empty\\.").containsMatchIn(prompt))
    }

    @Test fun anExistingFileKeepsEditsTargeted() {
        val prompt = AgentTurnPrompt.build("make the heading red", "dog.html", "dog.html", "<h1>Dogs</h1>", 16384,
            null, writeTool = "write_file", editTool = "replace_in_file")
        assertTrue(prompt.contains("```\n<h1>Dogs</h1>\n```"))
        assertTrue(prompt.contains("Use replace_in_file with an exact fragment"))
        assertFalse(prompt.contains("Target:"))
        assertFalse(prompt.contains("minimal stub"))
    }

    @Test fun withoutAnOpenFileItIsJustTheRequest() {
        assertEquals("hi", AgentTurnPrompt.build("hi", null, null, "", 16384, web, "Write", "Edit"))
    }
}
