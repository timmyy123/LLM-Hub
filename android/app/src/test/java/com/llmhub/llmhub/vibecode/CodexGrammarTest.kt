package com.llmhub.llmhub.vibecode

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class CodexGrammarTest {
    private val request = JSONObject("""{"input":"make a dog site","tools":[
        {"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"},"yield_time_ms":{"type":"number"}},"required":["cmd"]}},
        {"type":"function","name":"update_plan","parameters":{"type":"object","properties":{"plan":{"type":"array"}},"required":["plan"]}},
        {"type":"custom","name":"apply_patch"},
        {"type":"function","name":"list_mcp_resources","parameters":{"type":"object","properties":{}}}]}""")

    @Test fun allowsEveryOfferedToolWithItsRequiredArgumentsOnly() {
        val grammar = CodexGrammar.build(request)!!
        for (name in listOf("exec_command", "update_plan", "apply_patch", "list_mcp_resources", "write_file", "replace_in_file", "finish"))
            assertTrue(name, grammar.contains("\"\\\"$name\\\"\""))
        assertTrue(grammar.contains("\"\\\"cmd\\\"\" ws \":\" ws string"))
        assertFalse(grammar.contains("yield_time_ms"))
        assertTrue(grammar.contains("\"\\\"plan\\\"\" ws \":\" ws array"))
        assertTrue(grammar.contains("\"\\\"input\\\"\" ws \":\" ws string"))
        assertTrue(grammar.contains("\"\\\"path\\\"\" ws \":\" ws string ws \",\" ws \"\\\"old_text\\\"\""))
    }
}
