package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject

/** Parse model-emitted function arguments as data, preserving quoted shell scripts verbatim. */
internal object CodexToolArguments {
    private val assignment = Regex("[A-Za-z_][A-Za-z0-9_]*\\s*=")

    fun parse(source: String): JSONObject {
        val input = source.trim()
        val result = JSONObject()
        if (input.isEmpty()) return result
        if (assignment.find(input)?.range?.first != 0) {
            val value = if (input.first() in "\"'") {
                val (decoded, end) = string(input, 0)
                require(input.substring(end).isBlank()) { "Unexpected text after positional argument" }
                decoded
            } else input
            return result.put("path", value).put("command", value)
        }
        var position = 0
        while (position < input.length) {
            while (position < input.length && input[position].isWhitespace()) position++
            val key = assignment.find(input, position)
            require(key != null && key.range.first == position) { "Invalid tool argument at $position" }
            val name = key.value.substringBefore('=').trim()
            require(!result.has(name)) { "Duplicate tool argument: $name" }
            position = key.range.last + 1
            while (position < input.length && input[position].isWhitespace()) position++
            require(position < input.length) { "Missing tool argument value: $name" }
            val value: Any
            if (input[position] in "\"'") {
                val parsed = string(input, position)
                value = parsed.first
                position = parsed.second
            } else {
                val start = position
                val stack = mutableListOf<Char>()
                while (position < input.length) {
                    val c = input[position]
                    if (c in "\"'") { position = string(input, position).second; continue }
                    if (c in "[{(") stack.add(c)
                    else if (c in "]})") {
                        require(stack.isNotEmpty() && matches(stack.removeAt(stack.lastIndex), c)) { "Unbalanced tool argument" }
                    } else if (c == ',' && stack.isEmpty()) break
                    position++
                }
                require(stack.isEmpty()) { "Unclosed tool argument" }
                val literal = input.substring(start, position).trim()
                require(literal.isNotEmpty()) { "Missing tool argument value: $name" }
                value = when {
                    literal.equals("true", true) -> true
                    literal.equals("false", true) -> false
                    literal in listOf("None", "null") -> JSONObject.NULL
                    literal.startsWith('{') -> JSONObject(literal)
                    literal.startsWith('[') -> JSONArray(literal)
                    literal.toLongOrNull() != null -> literal.toLong()
                    literal.toDoubleOrNull() != null -> literal.toDouble()
                    else -> literal
                }
            }
            result.put(name, value)
            while (position < input.length && input[position].isWhitespace()) position++
            if (position == input.length) break
            require(input[position] == ',') { "Expected comma after tool argument: $name" }
            position++
            if (input.substring(position).isBlank()) break
        }
        return result
    }

    /** Find bracket calls without ending at brackets inside JavaScript or quoted file content. */
    fun bracketRanges(text: String): List<IntRange> = buildList {
        val beginning = Regex("\\[[A-Za-z_][A-Za-z0-9_.-]*\\s*(?:\\(|,)")
        var cursor = 0
        while (cursor < text.length) {
            val start = beginning.find(text, cursor)?.range?.first ?: break
            var end = start
            val stack = mutableListOf<Char>()
            try {
                while (end < text.length) {
                    val c = text[end]
                    if (c in "\"'") { end = string(text, end).second; continue }
                    if (c in "[{(") stack.add(c)
                    else if (c in "]})") {
                        require(stack.isNotEmpty() && matches(stack.removeAt(stack.lastIndex), c))
                        if (stack.isEmpty()) break
                    }
                    end++
                }
                require(end < text.length && stack.isEmpty()) { "Unclosed bracket tool call" }
                add(start..end)
                cursor = end + 1
            } catch (_: IllegalArgumentException) { cursor = start + 1 }
        }
    }

    private fun matches(open: Char, close: Char) = "[{(".indexOf(open) == "]})".indexOf(close)

    private fun string(input: String, start: Int): Pair<String, Int> {
        val quote = input[start]
        val delimiter = if (input.startsWith(quote.toString().repeat(3), start)) quote.toString().repeat(3) else quote.toString()
        var position = start + delimiter.length
        val value = StringBuilder()
        while (position < input.length) {
            if (input.startsWith(delimiter, position)) return value.toString() to position + delimiter.length
            val c = input[position++]
            if (c != '\\') { value.append(c); continue }
            require(position < input.length) { "Unclosed escape in tool argument" }
            when (val escaped = input[position++]) {
                '\\', '\'', '"', '/' -> value.append(escaped)
                'n' -> value.append('\n')
                'r' -> value.append('\r')
                't' -> value.append('\t')
                'b' -> value.append('\b')
                'f' -> value.append('\u000C')
                '\n' -> Unit
                'u', 'x' -> {
                    val length = if (escaped == 'u') 4 else 2
                    require(position + length <= input.length) { "Incomplete Unicode escape in tool argument" }
                    val code = input.substring(position, position + length).toIntOrNull(16)
                    require(code != null) { "Invalid Unicode escape in tool argument" }
                    value.append(code.toChar()); position += length
                }
                else -> value.append('\\').append(escaped)
            }
        }
        throw IllegalArgumentException("Unclosed quoted tool argument")
    }
}
