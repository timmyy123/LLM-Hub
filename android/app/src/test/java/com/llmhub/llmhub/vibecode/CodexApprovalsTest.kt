package com.llmhub.llmhub.vibecode

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class CodexApprovalsTest {
    @Test fun automaticallyAcceptsCommandsAndFileChanges() {
        for (method in listOf("item/commandExecution/requestApproval", "item/fileChange/requestApproval")) {
            assertEquals("accept", CodexApprovals.response(method, JSONObject())!!.getString("decision"))
        }
        assertEquals("never", CodexApprovals.POLICY)
        assertTrue(CodexConfig.arguments("http://127.0.0.1:8080/v1", 8192).contains("approval_policy="))
    }

    @Test fun grantsRequestedPermissionsForWholeSession() {
        val permissions = JSONObject("""{"network":{"enabled":true},"fileSystem":{"write":["/storage/emulated/0/Code"]}}""")
        val response = CodexApprovals.response("item/permissions/requestApproval", JSONObject().put("permissions", permissions))!!
        assertEquals("session", response.getString("scope"))
        assertEquals(permissions.toString(), response.getJSONObject("permissions").toString())
        assertNull(CodexApprovals.response("item/tool/requestUserInput", JSONObject()))
    }
}
