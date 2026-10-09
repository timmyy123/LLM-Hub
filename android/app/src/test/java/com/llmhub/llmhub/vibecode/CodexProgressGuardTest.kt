package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class CodexProgressGuardTest {
    @Test fun successfulEditInvalidatesPreviousFileContentsBeforeVerificationRead() {
        val req = request()
        req.getJSONArray("input").put(call("node successful-edit.js", "edit"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "edit").put("output", "Process exited with code 0"))
        assertTrue(CodexProgressGuard.supersededResults(req).contains("read"))
    }
    @Test(expected = IllegalArgumentException::class) fun readingAloneCannotCompleteAnEditingRequest() {
        val req = request()
        val input = JSONArray().put(JSONObject().put("type", "message").put("role", "user").put("content",
            "Active editor file: index.html\nRead the current file from disk using tools.\n\nUser request:\nuse tailwind css"))
        val original = req.getJSONArray("input")
        for(i in 0 until original.length()) input.put(original.get(i))
        req.put("input",input)
        CodexProgressGuard.validate(JSONArray().put(JSONObject().put("type", "message")),req)
    }
    @Test fun aTargetedSuccessfulCheckCanCompleteAnAlreadySatisfiedRequest() {
        val req = request()
        req.put("input",JSONArray().put(JSONObject().put("type", "message").put("role", "user").put("content", "use tailwind css"))
            .put(call("grep tailwind index.html", "verify"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "verify").put("output", "Process exited with code 0\n<script src=tailwind></script>")))
        CodexProgressGuard.validate(JSONArray().put(JSONObject().put("type", "message")),req)
    }
    @Test(expected = IllegalArgumentException::class) fun failedEditDoesNotAuthorizeReadingSameUnchangedFileAgain() {
        val req = request()
        req.getJSONArray("input").put(call("node failed-edit.js", "failed"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "failed")
                .put("output", "Process exited with code 1\nExact text not found; no file was changed"))
        CodexProgressGuard.validate(JSONArray().put(call("cat index.html")), req)
    }
    @Test(expected = IllegalArgumentException::class) fun identicalFailedEditIsNotExecutedAgain() {
        val req = request()
        req.getJSONArray("input").put(call("node failed-edit.js", "failed"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "failed")
                .put("output", "Process exited with code 1\nExact text not found"))
        CodexProgressGuard.validate(JSONArray().put(call("node failed-edit.js")), req)
    }
    @Test fun truncatedReadMayBeRetriedAndDoesNotReplaceFullSnapshot() {
        val req = request()
        req.getJSONArray("input").put(call("cat another.html", "partial"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "partial")
                .put("output", "Process exited with code 0\nWarning: truncated output\nOutput:\n<head>"))
        CodexProgressGuard.validate(JSONArray().put(call("cat another.html")), req)
        assertTrue(CodexProgressGuard.supersededResults(req).isEmpty())
    }
    private fun call(cmd: String, id: String = "read") = JSONObject().put("type", "function_call")
        .put("name", "exec_command").put("call_id", id)
        .put("arguments", JSONObject().put("cmd", cmd).toString())
    private fun request(cmd: String = "cat index.html", result: String = "Process exited with code 0\nOutput:\nrandomAffation") =
        JSONObject().put("tools", JSONArray().put(JSONObject().put("type", "function").put("name", "exec_command")
            .put("parameters", JSONObject().put("required", JSONArray().put("cmd")))))
            .put("input", JSONArray().put(call(cmd)).put(JSONObject().put("type", "function_call_output")
                .put("call_id", "read").put("output", result)))

    @Test(expected = IllegalArgumentException::class) fun rejectsRepeatingSuccessfulRead() {
        CodexProgressGuard.validate(JSONArray().put(call("cat index.html")), request())
    }
    @Test fun permitsEditAfterReadAndVerificationAfterEdit() {
        CodexProgressGuard.validate(JSONArray().put(call("sed -i 's/randomAffation/randomAffirmation/g' index.html")), request())
        CodexProgressGuard.validate(JSONArray().put(call("cat index.html")), request("sed -i 's/a/b/g' index.html"))
    }
    @Test fun allowsReadOnNewUserTurnAndRetryAfterFailure() {
        val req = request()
        req.getJSONArray("input").put(JSONObject().put("role", "user").put("content", "Read the file again"))
        CodexProgressGuard.validate(JSONArray().put(call("cat index.html")), req)
        CodexProgressGuard.validate(JSONArray().put(call("cat index.html")), request(result = "Process exited with code 1\nPermission denied"))
    }
    @Test fun promptEndsWithLatestExecutedResultAndPermitsCompletionAfterChecks() {
        val prompt = CodexResponses.prompt(request())
        assertTrue(prompt.substringAfterLast("Most recent tool result").contains("randomAffation"))
        assertTrue(prompt.contains("After successful verification, finish"))
        assertFalse(prompt.contains("cat gay.html"))
    }
    private fun verifiedEdit(): JSONObject = request().apply {
        getJSONArray("input")
            .put(call("sed -i 's/randomAffation/randomAffirmation/g' index.html", "edit"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "edit").put("output", "Process exited with code 0"))
            .put(call("cat \"index.html\"", "verify"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "verify")
                .put("output", "Process exited with code 0\nOutput:\ndisplay.textContent = randomAffirmation;"))
    }
    @Test fun supersededFileContentsAreRemovedFromModelPrompt() {
        val prompt = CodexResponses.prompt(verifiedEdit())
        assertFalse(prompt.contains("Output:\nrandomAffation"))
        assertTrue(prompt.contains("display.textContent = randomAffirmation;"))
        assertTrue(CodexProgressGuard.supersededResults(verifiedEdit()).contains("read"))
    }
    @Test(expected = IllegalArgumentException::class) fun rejectsEditReadEditLoopAfterSuccessfulVerification() {
        CodexProgressGuard.validate(JSONArray().put(call("sed -i 's/randomAffation/randomAffirmation/g' index.html")), verifiedEdit())
    }
    @Test fun permitsDifferentEditAfterVerification() {
        CodexProgressGuard.validate(JSONArray().put(call("sed -i 's/Other/Change/g' index.html")), verifiedEdit())
    }
    @Test fun recoveryRetainsExecutedEditAndLatestVerificationWithoutStaleSnapshots() {
        val prompt = CodexResponses.recoveryPrompt(verifiedEdit(), IllegalArgumentException("Repeated inspection"))
        assertTrue(prompt.contains("sed -i"))
        assertTrue(prompt.contains("display.textContent = randomAffirmation;"))
        assertFalse(prompt.contains("Output:\nrandomAffation"))
        assertTrue(prompt.contains("finish now"))
        assertTrue(prompt.contains("were NOT executed"))
    }
    @Test(expected = IllegalArgumentException::class) fun rejectsAlternatingInspectionsWithoutAnInterveningChange() {
        val req = request()
        req.getJSONArray("input").put(call("grep missing index.html", "search"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "search").put("output", "Process exited with code 1"))
        CodexProgressGuard.validate(JSONArray().put(call("cat \"index.html\"")), req)
    }
    @Test fun failedSearchDoesNotHideLatestVerifiedFileDuringRecovery() {
        val req = verifiedEdit()
        req.getJSONArray("input").put(call("grep randomAffation index.html", "search"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "search").put("output", "Process exited with code 1"))
        val prompt = CodexResponses.recoveryPrompt(req, IllegalArgumentException("Repeated edit"))
        assertTrue(prompt.contains("display.textContent = randomAffirmation;"))
        assertFalse(prompt.contains("Output:\nrandomAffation"))
    }
    @Test fun executedToolResultsBecomeSubsequentModelChatTurns() {
        val prompt = CodexResponses.prompt(verifiedEdit())
        assertTrue(prompt.startsWith("system: "))
        assertTrue(prompt.contains("\n\nassistant: Tool call already submitted:"))
        assertTrue(prompt.contains("\n\nuser: Executed tool result:"))
        assertTrue(prompt.substringAfterLast("\n\nuser: ").contains("Most recent tool result"))
    }
    @Test fun processInputToolIsOfferedOnlyWhenACommandIsStillRunning() {
        val req = request()
        req.getJSONArray("tools").put(JSONObject().put("type", "function").put("name", "write_stdin"))
        assertFalse(CodexResponses.prompt(req).contains("\"name\":\"write_stdin\""))
        req.getJSONArray("input").getJSONObject(1).put("output", "Process running with session ID 123")
        assertTrue(CodexResponses.prompt(req).contains("\"name\":\"write_stdin\""))
    }
}
