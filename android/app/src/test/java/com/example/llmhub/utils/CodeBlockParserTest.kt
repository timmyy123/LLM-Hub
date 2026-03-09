package com.example.llmhub.utils

import org.junit.Assert.*
import org.junit.Test

class CodeBlockParserTest {
    @Test
    fun `parses fenced block with language`() {
        val md = """Here is code:\n```python\nprint(1)\n```\nend"""
        val blocks = CodeBlockParser.parse(md)
        assertEquals(1, blocks.size)
        val b = blocks[0]
        assertEquals("python", b.language)
        assertEquals("print(1)\n", b.content)
        assertFalse(b.isInline)
    }

    @Test
    fun `parses fenced block without language`() {
        val md = "Start\n```\nlet x = 1\n```\n"
        val blocks = CodeBlockParser.parse(md)
        assertEquals(1, blocks.size)
        assertNull(blocks[0].language)
        assertEquals("let x = 1\n", blocks[0].content)
    }

    @Test
    fun `parses multiple blocks and inline`() {
        val md = "Code: `inline` and:\n```js\nconsole.log('hi')\n```\nMore `x`"
        val blocks = CodeBlockParser.parse(md)
        assertEquals(3, blocks.size)
        assertTrue(blocks[0].isInline)
        assertEquals("inline", blocks[0].content)
        assertEquals("js", blocks[1].language)
        assertEquals("console.log('hi')\n", blocks[1].content)
        assertTrue(blocks[2].isInline)
        assertEquals("x", blocks[2].content)
    }
}
