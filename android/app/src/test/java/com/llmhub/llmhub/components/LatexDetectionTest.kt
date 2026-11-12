package com.llmhub.llmhub.components

import org.junit.Assert.assertEquals
import org.junit.Test

/** Which parts of a reply the chat bubble hands to the KaTeX WebView. Prices must never be among them. */
class LatexDetectionTest {
    private fun math(text: String) = latexRegex.findAll(text).map { it.value }.toList()

    @Test fun pricesAreNotMath() {
        assertEquals(emptyList<String>(), math("Estimated AI subtotal \$0.18-\$0.56 before telephone. The US\$0.10 Nova-3 line is high."))
        assertEquals(emptyList<String>(), math("Plan A costs \$5 and plan B costs \$10 per month."))
        assertEquals(emptyList<String>(), math("| Vapi | \$0.05/min | \$0.07/min |\n| Retell | \$0.07 | \$0.10 |"))
        assertEquals(emptyList<String>(), math("It was \$5 for one and I still had 10\$ left."))
        assertEquals(emptyList<String>(), math("Total: \$100 - \$50 = \$50"))
        assertEquals(emptyList<String>(), math("One line ends with a price of \$3\nand the next starts with \$4 more."))
    }

    @Test fun inlineMathIsStillMath() {
        assertEquals(listOf("\$x^2 + y^2 = z^2\$"), math("By Pythagoras \$x^2 + y^2 = z^2\$ holds."))
        assertEquals(listOf("\$2x + 3 = 7\$"), math("Solve \$2x + 3 = 7\$ for x."))
        assertEquals(listOf("\$5\$", "\$n\$"), math("The answer is \$5\$ when \$n\$ is odd."))
        assertEquals(listOf("\$\\frac{a}{b}\$"), math("The ratio \$\\frac{a}{b}\$ is fixed."))
    }

    @Test fun blockAndParenMathIsStillMath() {
        assertEquals(listOf("\$\$E = mc^2\$\$"), math("Energy:\n\$\$E = mc^2\$\$\ndone"))
        assertEquals(listOf("\\[a^2\\]", "\\(b\\)"), math("First \\[a^2\\] then \\(b\\)."))
    }

    /** While a reply streams in, no moment of it may look like math: each one would flash a WebView. */
    @Test fun pricesAreNotMathAtAnyPointWhileStreaming() {
        val reply = "- **Planning allowance**: \$0.18 – \$0.56 (estimated AI subtotal).\n" +
            "- **US\$0.10 Nova-3 line**: high for STT alone.\n" +
            "- Voice Live Lite/Standard (\$0.025–\$0.07/min) and Pro (\$0.07–\$0.15/min).\n" +
            "The subtotal is **\$0.18–\$0.56**, with per-minute charges on **Voice Live** (\$0.025–\$0.07)."
        for (end in 1..reply.length) {
            assertEquals("after ${reply.substring(0, end).takeLast(30)}", emptyList<String>(), math(reply.substring(0, end)))
        }
    }

    @Test fun mathAtTheEndOfAReplyIsStillMath() {
        assertEquals(listOf("\$2x + 3 = 7\$"), math("Solve \$2x + 3 = 7\$"))
        assertEquals(listOf("\$x\$"), math("The unknown is \$x\$"))
    }
}
