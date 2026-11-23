package com.llmhub.llmhub.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class InklingFormatTest {
    private val open = "[T]"
    private val close = "[/T]"

    private fun parse(tokens: List<String>, thinking: Boolean): String {
        val parser = InklingOutputParser(open, close)
        parser.reset(thinking)
        val out = StringBuilder()
        tokens.forEach { parser.accept(it) { s -> out.append(s) } }
        parser.finish { out.append(it) }
        return out.toString()
    }

    private val reply = listOf(
        "<|message_model|>", "<|content_thinking|>", "The user", " asks.", "<|end_message|>",
        "<|message_model|>", "<|content_text|>", "Paris", ".", "<|end_message|>", "<|content_model_end_sampling|>",
    )

    @Test
    fun thinkingOnShowsThoughtThenAnswer() {
        assertEquals("[T]The user asks.[/T]Paris.", parse(reply, thinking = true))
    }

    @Test
    fun thinkingOffStillShowsWhatTheModelThought() {
        assertEquals("[T]The user asks.[/T]Paris.", parse(reply, thinking = false))
    }

    @Test
    fun replyStoppedInsideThoughtClosesIt() {
        assertEquals("[T]Still thinking[/T]", parse(listOf("<|message_model|>", "<|content_thinking|>", "Still thinking"), thinking = true))
    }

    @Test
    fun emptyThoughtShowsNoBlock() {
        val tokens = listOf("<|message_model|>", "<|content_thinking|>", "<|end_message|>", "<|message_model|>", "<|content_text|>", "Hi", "<|end_message|>")
        assertEquals("Hi", parse(tokens, thinking = true))
    }

    @Test
    fun promptMatchesTemplate() {
        val prompt = InklingFormat.buildPrompt(
            listOf(
                VlmPromptTurn("system", "Be brief."),
                VlmPromptTurn("user", "Hi"),
                VlmPromptTurn("assistant", "Hello!"),
                VlmPromptTurn("user", "<__media__>\nWhat is this?"),
            ),
            thinkingEnabled = true,
            mediaMarker = "<__media__>",
            answerOf = { it },
        )
        assertEquals(
            "<|message_system|><|content_text|>Be brief.<|end_message|>" +
                "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" +
                "<|message_user|><|content_text|>Hi<|end_message|>" +
                "<|message_model|><|content_text|>Hello!<|end_message|><|content_model_end_sampling|>" +
                "<|message_user|><__media__><|end_message|>" +
                "<|message_user|><|content_text|>What is this?<|end_message|>",
            prompt,
        )
    }

    @Test
    fun thinkingLevelsAreWrittenLikeTheTemplate() {
        listOf(0.1f to "0.1", 0.2f to "0.2", 0.7f to "0.7", 0.9f to "0.9", 0.99f to "0.99").forEach { (effort, text) ->
            val prompt = InklingFormat.buildPrompt(listOf(VlmPromptTurn("user", "Hi")), true, "<__media__>", effort) { it }
            assertEquals(true, prompt.startsWith("<|message_system|><|content_text|>Thinking effort level: $text<|end_message|>"))
        }
    }

    @Test
    fun thinkingOffPromptUsesEffortZero() {
        val prompt = InklingFormat.buildPrompt(listOf(VlmPromptTurn("user", "Hi")), false, "<__media__>") { it }
        assertEquals(
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hi<|end_message|>",
            prompt,
        )
    }
}
