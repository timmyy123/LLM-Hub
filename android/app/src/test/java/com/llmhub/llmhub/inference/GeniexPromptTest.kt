package com.llmhub.llmhub.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GeniexPromptTest {
    @Test
    fun simpleUserPromptRemainsSimple() {
        assertEquals("Hello", prepareVlmUserText("user: Hello\nassistant:"))
    }

    @Test
    fun ragMemoryAfterUserMessageIsPreserved() {
        val prompt = """
            user: What is my favourite colour?

            ---

            USER MEMORY FACTS:

            The user's favourite colour is green.

            ---

            assistant:
        """.trimIndent()

        val result = prepareVlmUserText(prompt)

        assertTrue(result.startsWith("What is my favourite colour?"))
        assertTrue(result.contains("USER MEMORY FACTS:"))
        assertTrue(result.contains("favourite colour is green"))
        assertTrue(!result.endsWith("assistant:"))
    }

    @Test
    fun conversationRolesAndSystemPromptArePreserved() {
        val prompt = """
            system: Be concise.

            user: My name is Sam.

            assistant: Nice to meet you.

            user: What is my name?
            assistant:
        """.trimIndent()

        val result = prepareVlmUserText(prompt)

        assertTrue(result.startsWith("system: Be concise."))
        assertTrue(result.contains("assistant: Nice to meet you."))
        assertTrue(result.contains("user: What is my name?"))
    }

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
}
