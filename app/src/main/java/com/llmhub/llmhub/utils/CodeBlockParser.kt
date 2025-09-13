package com.example.llmhub.utils

import com.vladsch.flexmark.ast.Code
import com.vladsch.flexmark.ast.FencedCodeBlock
import com.vladsch.flexmark.parser.Parser
import com.vladsch.flexmark.util.ast.Node

/**
 * Parser that extracts fenced code blocks (with optional language) and inline code
 * from Markdown text using Flexmark.
 */
object CodeBlockParser {
    private val parser: Parser = Parser.builder().build()

    fun parse(text: String): List<CodeBlock> {
        // Some model UIs use a single backtick on its own line to delimit code blocks.
        // Normalize those to standard fenced code blocks (``` ) so Flexmark parses them.
        val normalized = normalizeSingleBacktickBlocks(text)
        val document = parser.parse(normalized)
        val result = mutableListOf<CodeBlock>()

        fun walk(node: Node?) {
            var child = node?.firstChild
            while (child != null) {
                when (child) {
                    is FencedCodeBlock -> {
                        val info = child.info?.toString()?.trim() ?: ""
                        val lang = info.takeIf { it.isNotBlank() }?.split(Regex("\\s+"))?.firstOrNull()
                        val content = child.contentChars?.toString() ?: ""
                        result.add(CodeBlock(lang, content, isInline = false))
                    }
                    is Code -> {
                        val literal = child.chars?.toString() ?: ""
                        result.add(CodeBlock(null, literal, isInline = true))
                    }
                }
                walk(child)
                child = child.next
            }
        }

        walk(document)
        return result
    }

    private fun normalizeSingleBacktickBlocks(input: String): String {
        var s = input
        // Replace lines that are exactly "`lang" with "```lang" (opening)
        val langOpen = Regex("(?m)^\\s*`([A-Za-z0-9_+\\-]+)\\s*$")
        s = langOpen.replace(s) { match -> "```${match.groupValues[1]}" }

        // Replace lines that are exactly a single backtick with triple backticks (closing or generic fence)
        val bareOpen = Regex("(?m)^\\s*`\\s*$")
        s = bareOpen.replace(s) { "```" }

        return s
    }

    sealed class ParsedSegment {
        data class Text(val text: String) : ParsedSegment()
        data class Code(val language: String?, val content: String) : ParsedSegment()
    }

    /**
     * Splits the input into a list of text/code segments preserving order.
     * This is intended for UI rendering where code blocks should appear inline
     * in the original text flow.
     */
    fun parseSegments(text: String): List<ParsedSegment> {
        val normalized = normalizeSingleBacktickBlocks(text)
        val pattern = Regex("(?s)```\\s*([A-Za-z0-9_+\\-]*)\\s*\\n(.*?)\\n?```")
        val result = mutableListOf<ParsedSegment>()
        var lastIndex = 0

        for (m in pattern.findAll(normalized)) {
            if (m.range.first > lastIndex) {
                val before = normalized.substring(lastIndex, m.range.first)
                if (before.isNotBlank()) result.add(ParsedSegment.Text(before))
            }
            val lang = m.groupValues.getOrNull(1)?.ifBlank { null }
            val content = m.groupValues.getOrNull(2) ?: ""
            result.add(ParsedSegment.Code(lang, content))
            lastIndex = m.range.last + 1
        }

        if (lastIndex < normalized.length) {
            val tail = normalized.substring(lastIndex)
            if (tail.isNotBlank()) result.add(ParsedSegment.Text(tail))
        }

        // If no segments found, return the whole text as a Text segment
        if (result.isEmpty()) return listOf(ParsedSegment.Text(text))
        return result
    }
}
