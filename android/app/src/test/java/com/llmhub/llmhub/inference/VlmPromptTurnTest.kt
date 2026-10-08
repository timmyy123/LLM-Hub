package com.llmhub.llmhub.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VlmPromptTurnTest {
    @Test
    fun transcriptIsSplitIntoRealVlmRolesAndRagStaysWithLatestUser() {
        val prompt = """
            system: Be concise.

            user: My name is Sam.

            assistant: Nice to meet you.

            user: What is my name?

            USER MEMORY FACTS:
            The user's name is Sam.

            assistant:
        """.trimIndent()

        val turns = parseVlmPromptTurns(prompt)

        assertEquals(listOf("system", "user", "assistant", "user"), turns.map { it.role })
        assertEquals("Be concise.", turns[0].text)
        assertEquals("Nice to meet you.", turns[2].text)
        assertTrue(turns[3].text.startsWith("What is my name?"))
        assertTrue(turns[3].text.contains("The user's name is Sam."))
    }

    @Test
    fun multiTurnVlmTranscriptIsProperlyParsedForImageFollowUp() {
        val prompt = """
            user: 📄 563.jpg
            assistant: The image shows a white minivan parked outdoors.
            user: what
            assistant:
        """.trimIndent()

        val turns = parseVlmPromptTurns(prompt)
        assertEquals(listOf("user", "assistant", "user"), turns.map { it.role })
        assertEquals("📄 563.jpg", turns[0].text)
        assertEquals("The image shows a white minivan parked outdoors.", turns[1].text)
        assertEquals("what", turns[2].text)
    }
}
