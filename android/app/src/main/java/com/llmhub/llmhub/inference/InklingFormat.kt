package com.llmhub.llmhub.inference

/**
 * Inkling 2 (Thinking Machines) chat format, written like the GGUF's chat template (llama.cpp's built-in template
 * list has no Inkling format). Every message is a block closed by <|end_message|>:
 *   <|message_user|><|content_text|>...<|end_message|>
 *   <|message_model|><|content_thinking|>...<|end_message|><|message_model|><|content_text|>...<|end_message|>
 * Thinking is set by a system block, "Thinking effort level: <0..0.99>", before the first non-system message. The
 * template has no generation prompt: the model opens its own <|message_model|> block.
 */
internal object InklingFormat {
    const val MODEL = "<|message_model|>"
    const val USER = "<|message_user|>"
    const val SYSTEM = "<|message_system|>"
    const val THINK = "<|content_thinking|>"
    const val TEXT = "<|content_text|>"
    const val END = "<|end_message|>"
    const val END_SAMPLING = "<|content_model_end_sampling|>"

    /** The template's default effort ("high") with thinking on, 0 ("none") with it off. */
    private fun effortBlock(thinkingEnabled: Boolean) =
        SYSTEM + TEXT + "Thinking effort level: " + (if (thinkingEnabled) "0.9" else "0") + END

    /**
     * [answerOf] turns an earlier reply into the answer alone (its thinking is not fed back). An image marker in a
     * user turn becomes its own user block, as llama.cpp's Inkling handler splits it: mtmd opens the image block.
     */
    fun buildPrompt(
        messages: List<VlmPromptTurn>,
        thinkingEnabled: Boolean,
        mediaMarker: String,
        answerOf: (String) -> String,
    ): String {
        val sb = StringBuilder()
        var effortWritten = false
        fun writeEffort() {
            if (!effortWritten) {
                sb.append(effortBlock(thinkingEnabled))
                effortWritten = true
            }
        }
        messages.forEach { turn ->
            when (turn.role) {
                "system" -> turn.text.trim().takeIf { it.isNotEmpty() }?.let { sb.append(SYSTEM).append(TEXT).append(it).append(END) }
                "assistant" -> {
                    writeEffort()
                    val answer = answerOf(turn.text)
                    if (answer.isNotEmpty()) sb.append(MODEL).append(TEXT).append(answer).append(END).append(END_SAMPLING)
                }
                else -> {
                    writeEffort()
                    appendUser(sb, turn.text, mediaMarker)
                }
            }
        }
        writeEffort()
        return sb.toString()
    }

    private fun appendUser(sb: StringBuilder, text: String, mediaMarker: String) {
        var rest = text
        while (true) {
            val at = if (mediaMarker.isEmpty()) -1 else rest.indexOf(mediaMarker)
            val part = (if (at >= 0) rest.substring(0, at) else rest).trim()
            if (part.isNotEmpty()) sb.append(USER).append(TEXT).append(part).append(END)
            if (at < 0) break
            sb.append(USER).append(mediaMarker).append(END)
            rest = rest.substring(at + mediaMarker.length)
        }
    }
}

/**
 * Streams Inkling output as the app's text: thinking blocks between [thinkOpen] and [thinkClose] when thinking is
 * on, dropped when it is off (effort 0 does not fully stop the model reasoning), and the text blocks as the answer.
 */
internal class InklingOutputParser(
    private val thinkOpen: String,
    private val thinkClose: String,
) {
    private enum class State { BETWEEN, IN_THINK, IN_TEXT }

    private val buffer = StringBuilder()
    private var state = State.BETWEEN
    private var thinkingEnabled = true
    private var thinkingOpened = false

    fun reset(thinkingEnabled: Boolean) {
        buffer.clear()
        state = State.BETWEEN
        this.thinkingEnabled = thinkingEnabled
        thinkingOpened = false
    }

    fun accept(text: String, send: (String) -> Unit) {
        buffer.append(text)
        while (true) {
            when (state) {
                State.BETWEEN -> {
                    val value = stripMarkers(buffer.toString())
                    val think = value.indexOf(InklingFormat.THINK)
                    val answer = value.indexOf(InklingFormat.TEXT)
                    val first = listOf(think, answer).filter { it >= 0 }.minOrNull()
                    if (first != null) {
                        // Text before any content token is answer text the model wrote without a block.
                        value.substring(0, first).takeIf { it.isNotBlank() }?.let(send)
                        state = if (first == think) State.IN_THINK else State.IN_TEXT
                        val token = if (first == think) InklingFormat.THINK else InklingFormat.TEXT
                        buffer.clear().append(value.substring(first + token.length))
                        continue
                    }
                    val leading = value.trimStart()
                    if (leading.isEmpty() || (leading.startsWith("<|") && !leading.contains("|>"))) {
                        buffer.clear().append(value)
                        return
                    }
                    // Any other block (a tool call) is passed on as it is.
                    state = State.IN_TEXT
                    buffer.clear().append(value)
                }
                State.IN_THINK -> {
                    val value = buffer.toString()
                    val close = earliest(value, InklingFormat.END, InklingFormat.MODEL, InklingFormat.END_SAMPLING)
                    if (close != null) {
                        sendThought(value.substring(0, close.first), send)
                        if (thinkingOpened) send(thinkClose)
                        thinkingOpened = false
                        buffer.clear().append(value.substring(close.first + close.second.length))
                        state = State.BETWEEN
                        continue
                    }
                    val safe = safeLength(value)
                    if (safe > 0) {
                        sendThought(value.substring(0, safe), send)
                        buffer.delete(0, safe)
                    }
                    return
                }
                State.IN_TEXT -> {
                    val value = buffer.toString()
                    val close = earliest(value, InklingFormat.END, InklingFormat.MODEL, InklingFormat.END_SAMPLING)
                    if (close != null) {
                        value.substring(0, close.first).takeIf { it.isNotEmpty() }?.let(send)
                        buffer.clear().append(value.substring(close.first + close.second.length))
                        state = State.BETWEEN
                        continue
                    }
                    val safe = safeLength(value)
                    if (safe > 0) {
                        send(value.substring(0, safe))
                        buffer.delete(0, safe)
                    }
                    return
                }
            }
        }
    }

    fun finish(send: (String) -> Unit) {
        val value = stripMarkers(buffer.toString())
        when (state) {
            State.IN_THINK -> {
                sendThought(value, send)
                if (thinkingOpened) send(thinkClose)
            }
            State.IN_TEXT -> if (value.isNotEmpty()) send(value)
            State.BETWEEN -> if (value.isNotBlank() && !value.trimStart().startsWith("<|")) send(value)
        }
        reset(thinkingEnabled)
    }

    /** Opens the thinking block on the first visible reasoning, so an empty thinking block shows nothing. */
    private fun sendThought(text: String, send: (String) -> Unit) {
        if (!thinkingEnabled) return
        if (!thinkingOpened) {
            val visible = text.trimStart()
            if (visible.isEmpty()) return
            thinkingOpened = true
            send(thinkOpen)
            send(visible)
        } else if (text.isNotEmpty()) {
            send(text)
        }
    }

    private fun stripMarkers(value: String) =
        value.replace(InklingFormat.MODEL, "").replace(InklingFormat.END_SAMPLING, "").replace(InklingFormat.END, "")

    private fun earliest(value: String, vararg tokens: String): Pair<Int, String>? =
        tokens.mapNotNull { token -> value.indexOf(token).takeIf { it >= 0 }?.let { it to token } }.minByOrNull { it.first }

    /** Holds back a trailing "<|..." that may still become a block token. */
    private fun safeLength(value: String): Int {
        val open = value.lastIndexOf("<|")
        return if (open >= 0 && !value.substring(open).contains("|>")) open else value.length
    }
}
