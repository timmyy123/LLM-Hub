package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject

/**
 * A llama.cpp GBNF grammar for the reply format in [CodexResponses.prompt], built from the tools offered in the
 * request: {"text": short string, "tool_calls": [calls]}, where each call names an offered tool and carries its
 * required arguments with their JSON types. Constrained decoding makes malformed JSON, invented tools and missing
 * arguments impossible, whatever the prompt or model; the short text keeps it from spending its reply budget narrating.
 */
internal object CodexGrammar {
    fun build(request: JSONObject): String? = forTools(CodexResponses.offeredTools(request))

    /**
     * [tools]: objects with name, type ("custom" takes raw input) and a JSON schema in parameters.
     * [requireCall]: at least one call, for a model that keeps ending without using any tool.
     * At most three calls: small models otherwise loop through the tool list until the reply limit cuts them off.
     */
    fun forTools(tools: JSONArray, requireCall: Boolean = false): String? {
        if (tools.length() == 0) return null
        val rules = StringBuilder()
        val calls = (0 until tools.length()).map { i ->
            val tool = tools.getJSONObject(i)
            val name = "\"\\\"" + tool.getString("name") + "\\\"\""
            val body = if (tool.optString("type") == "custom") "\"\\\"input\\\"\" ws \":\" ws string"
                else "\"\\\"arguments\\\"\" ws \":\" ws " + arguments(tool.optJSONObject("parameters"))
            rules.append("call$i ::= \"{\" ws \"\\\"name\\\"\" ws \":\" ws $name ws \",\" ws $body ws \"}\"\n")
            "call$i"
        }
        return """
            root ::= "{" ws "\"text\"" ws ":" ws note ws "," ws "\"tool_calls\"" ws ":" ws "[" ws ${if (requireCall) "call (ws \",\" ws call){0,2}" else "(call (ws \",\" ws call){0,2})?"} ws "]" ws "}"
            call ::= ${calls.joinToString(" | ")}
        """.trimIndent() + "\n" + rules + JSON_RULES
    }

    /** The required arguments, in schema order, each with its declared type. Optional ones are left out. */
    private fun arguments(parameters: JSONObject?): String {
        val properties = parameters?.optJSONObject("properties") ?: JSONObject()
        val required = parameters?.optJSONArray("required") ?: JSONArray()
        val fields = (0 until required.length()).map { required.getString(it) }.map { key ->
            val type = properties.optJSONObject(key)?.optString("type").orEmpty()
            "\"\\\"" + key + "\\\"\" ws \":\" ws " + when (type) {
                "string" -> "string"
                "integer" -> "integer"
                "number" -> "number"
                "boolean" -> "boolean"
                "array" -> "array"
                "object" -> "object"
                else -> "value"
            }
        }
        return if (fields.isEmpty()) "\"{\" ws \"}\"" else "\"{\" ws " + fields.joinToString(" ws \",\" ws ") + " ws \"}\""
    }

    private val JSON_RULES = """
        value ::= object | array | string | number | boolean | "null"
        object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
        array ::= "[" ws (value (ws "," ws value)*)? ws "]"
        string ::= "\"" char* "\""
        note ::= "\"" char{0,240} "\""
        char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
        integer ::= "-"? ([0-9] | [1-9] [0-9]{0,15})
        number ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
        boolean ::= "true" | "false"
        ws ::= [ \t\n]{0,8}
    """.trimIndent() + "\n"
}
