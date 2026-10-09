package com.llmhub.llmhub.vibecode

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files

class CodexFileEditsTest {
    @Test fun executedEditHistoryDoesNotRepeatEncodedFileContents() {
        val source = "<style>" + "body { color: pink; }".repeat(100) + "</style>"
        val cmd = CodexFileEdits.replaceCommand("index.html", source, source + "\nnew")
        val summary = CodexFileEdits.commandSummary(cmd)!!
        assertTrue(summary.contains("index.html"))
        assertTrue(summary.length < 200)
        assertFalse(summary.contains("body"))
    }
    private fun edit(source: String, old: String, replacement: String, extension: String = "html"): Pair<Int, String> {
        val dir = Files.createTempDirectory("codex-indentation-").toFile()
        try {
            val file = dir.resolve("test.$extension"); file.writeText(source)
            val process = ProcessBuilder("bash", "-c", CodexFileEdits.replaceCommand(file.path, old, replacement))
                .redirectErrorStream(true).start()
            process.inputStream.bufferedReader().readText()
            return process.waitFor() to file.readText()
        } finally { dir.deleteRecursively() }
    }
    @Test fun missingClosingTagIndentationDoesNotPreventActualEdit() {
        val old = "<style>\n        body { color: pink; }\n</style>"
        val source = "<head>\n    <style>\n        body { color: pink; }\n    </style>\n</head>"
        val result = edit(source, old, "<script src=\"https://cdn.tailwindcss.com\"></script>")
        assertEquals(0, result.first)
        assertEquals("<head>\n    <script src=\"https://cdn.tailwindcss.com\"></script>\n</head>", result.second)
    }
    @Test fun differingCodeIsNeverFuzzilyReplaced() {
        val source = "<style>\nbody { color: blue; }\n</style>"
        val result = edit(source, "<style>\nbody { color: pink; }\n</style>", "changed")
        assertNotEquals(0, result.first); assertEquals(source, result.second)
    }
    @Test fun ambiguousIndentationMatchLeavesFileUntouched() {
        val source = "<p>\n    Hello\n</p>\n<p>\n  Hello\n</p>"
        val result = edit(source, "<p>\nHello\n</p>", "changed")
        assertNotEquals(0, result.first); assertEquals(source, result.second)
    }
    @Test fun indentationSensitiveLanguagesRequireExactMatch() {
        val source = "if enabled:\n    run()"
        val result = edit(source, "if enabled:\nrun()", "changed", "py")
        assertNotEquals(0, result.first); assertEquals(source, result.second)
    }
    @Test fun noOpIsNotReportedAsAFileChange() {
        val result = edit("hello", "hello", "hello")
        assertNotEquals(0, result.first); assertEquals("hello", result.second)
    }
}
