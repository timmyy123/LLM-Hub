package com.llmhub.llmhub.vibecode

import java.net.HttpURLConnection
import java.net.URL
import org.junit.Assert.*
import org.junit.Test

class LocalCodexServerTest {
    @Test fun redundantReadAfterExactSavedFileVerificationCompletesWithoutAnotherCommand() {
        val content = "<html>pig</html>"
        val encoded = java.util.Base64.getEncoder().encodeToString(content.toByteArray())
        var attempts = 0
        val input = org.json.JSONArray()
            .put(org.json.JSONObject().put("type", "message").put("role", "user").put("content", "make a pig site"))
        for ((id, command, result) in listOf(
            Triple("write", "printf '%s' '$encoded' | base64 -d > 'pig.html'", "Process exited with code 0\nOutput:\n"),
            Triple("read", "cat pig.html", "Process exited with code 0\nOutput:\n$content"))) {
            input.put(org.json.JSONObject().put("type", "function_call").put("name", "exec_command").put("call_id", id)
                .put("arguments", org.json.JSONObject().put("cmd", command).toString()))
            input.put(org.json.JSONObject().put("type", "function_call_output").put("call_id", id).put("output", result))
        }
        LocalCodexServer(infer = { attempts++; """{"tool_calls":[{"name":"exec_command","arguments":{"cmd":"cat pig.html"}}]}""" },
            modelError = "Invalid", verifiedFileSummary = { "Saved and verified $it" }).use { server ->
            val request = org.json.JSONObject().put("input", input).put("tools", org.json.JSONArray()
                .put(org.json.JSONObject().put("type", "function").put("name", "exec_command")))
            val (_, stream) = post("${server.baseUrl}/responses", request.toString())
            assertEquals(1, attempts)
            assertTrue(stream.contains("response.completed"))
            assertTrue(stream.contains("Saved and verified pig.html"))
            assertFalse(stream.contains("response.failed"))
            assertFalse(stream.contains("function_call"))
        }
    }
    @Test fun nativeFinalAnswerAfterVerifiedEditDoesNotRestartTheAgent() {
        var attempts = 0
        LocalCodexServer(infer = { attempts++; CodexResponses.SENTINEL_THINK + "The edit is saved and verified. No further action is needed." }, modelError = "Invalid").use { server ->
            val (_, stream) = post("${server.baseUrl}/responses", """{"input":[
                {"type":"message","role":"user","content":"edit index.html"},
                {"type":"function_call","name":"exec_command","call_id":"edit","arguments":"{\"cmd\":\"node edit.js\"}"},
                {"type":"function_call_output","call_id":"edit","output":"Process exited with code 0"},
                {"type":"function_call","name":"exec_command","call_id":"read","arguments":"{\"cmd\":\"cat index.html\"}"},
                {"type":"function_call_output","call_id":"read","output":"Process exited with code 0\nOutput:\nupdated"}],"tools":[]}""")
            assertEquals(1,attempts)
            assertTrue(stream.contains("response.completed"))
            assertFalse(stream.contains("response.failed"))
        }
    }
    @Test fun nativeSuccessClaimWithoutAnEditOrCheckIsStillRejected() {
        LocalCodexServer(infer = { CodexResponses.SENTINEL_THINK + "All changes are complete." }, modelError = "Invalid").use { server ->
            val (_, stream) = post("${server.baseUrl}/responses", """{"input":[{"type":"message","role":"user","content":"edit index.html"}],"tools":[]}""")
            assertTrue(stream.contains("response.failed"))
            assertFalse(stream.contains("response.completed"))
        }
    }
    @Test fun formattingRecoveryDoesNotForgetRejectedCompletion() {
        var attempts = 0
        LocalCodexServer(infer = { prompt ->
            when (++attempts) {
                1 -> """{"text":"Already completed","tool_calls":[]}"""
                2 -> "<think>I should edit it</think>"
                else -> {
                    assertTrue(prompt.contains("Unverified completion"))
                    """{"tool_calls":[{"name":"exec_command","arguments":{"cmd":"grep tailwind index.html"}}]}"""
                }
            }
        }, modelError = "Invalid response").use { server ->
            val (_, stream) = post("${server.baseUrl}/responses",
                """{"input":[{"type":"message","role":"user","content":"Active editor file: index.html\n\nUser request:\nuse tailwind css"}],"tools":[{"type":"function","name":"exec_command","parameters":{"required":["cmd"]}}]}""")
            assertEquals(3, attempts)
            assertTrue(stream.contains("response.completed"))
            assertFalse(stream.contains("Already completed"))
        }
    }
    @Test fun recoveryBeforeAnyToolRanAsksForAToolInsteadOfAFinalReply() {
        var attempts = 0
        LocalCodexServer(infer = { prompt ->
            when (++attempts) {
                1 -> """{"text":"I've read the file dog.html and will now edit it.","tool_calls":[]}"""
                else -> {
                    assertTrue(prompt.contains("No tool has run yet"))
                    assertFalse(prompt.contains("finish now"))
                    """{"text":"Writing","tool_calls":[{"name":"exec_command","arguments":{"cmd":"ls"}}]}"""
                }
            }
        }, modelError = "Invalid response").use { server ->
            val (_, stream) = post("${server.baseUrl}/responses",
                """{"input":[{"type":"message","role":"user","content":"Active editor file: dog.html\n\nUser request:\nmake a dog site"}],"tools":[{"type":"function","name":"exec_command","parameters":{"required":["cmd"]}}]}""")
            assertEquals(2, attempts)
            assertTrue(stream.contains("response.completed"))
            assertFalse(stream.contains("will now edit it"))
        }
    }
    @Test fun successClaimRightAfterALiteralWriteReadsTheFileBackInsteadOfFailing() {
        val encoded = java.util.Base64.getEncoder().encodeToString("<h1>Dogs</h1>".toByteArray())
        var attempts = 0
        LocalCodexServer(infer = { attempts++; """{"text":"dog.html has been created.","tool_calls":[]}""" },
            modelError = "Invalid").use { server ->
            val request = org.json.JSONObject().put("input", org.json.JSONArray()
                .put(org.json.JSONObject().put("type", "message").put("role", "user")
                    .put("content", "Active editor file: dog.html\n\nUser request:\nmake a dog site"))
                .put(org.json.JSONObject().put("type", "function_call").put("name", "exec_command").put("call_id", "write")
                    .put("arguments", org.json.JSONObject().put("cmd", "printf '%s' '$encoded' | base64 -d > 'dog.html'").toString()))
                .put(org.json.JSONObject().put("type", "function_call_output").put("call_id", "write")
                    .put("output", "Process exited with code 0\nOutput:\n")))
                .put("tools", org.json.JSONArray().put(org.json.JSONObject().put("type", "function").put("name", "exec_command")
                    .put("parameters", org.json.JSONObject().put("required", org.json.JSONArray().put("cmd")))))
            val (_, stream) = post("${server.baseUrl}/responses", request.toString())
            assertEquals(1, attempts)
            assertFalse(stream.contains("response.failed"))
            val call = stream.lineSequence().filter { it.startsWith("data: ") }.map { org.json.JSONObject(it.removePrefix("data: ")) }
                .mapNotNull { it.optJSONObject("item") }.first { it.optString("type") == "function_call" }
            assertEquals("cat \"dog.html\"", org.json.JSONObject(call.getString("arguments")).getString("cmd"))
            assertFalse(stream.contains("has been created"))
        }
    }
    @Test fun theServersOwnSuccessfulReadBackEndsTheTurnWithoutAskingTheModel() {
        var attempts = 0
        LocalCodexServer(infer = { attempts++; """{"text":"I will now write the file again","tool_calls":[]}""" },
            modelError = "Invalid", verifiedFileSummary = { "Saved and verified $it" }).use { server ->
            val input = """[{"type":"message","role":"user","content":"change the heading color to red"},
                {"type":"function_call","name":"exec_command","call_id":"call_llmhub_readback_1","arguments":"{\"cmd\":\"cat \\\"dog.html\\\"\"}"},
                {"type":"function_call_output","call_id":"call_llmhub_readback_1","output":"Process exited with code 0\nOutput:\n<h1 style=\"color:red\">Dogs</h1>"}]"""
            val (_, stream) = post("${server.baseUrl}/responses", """{"input":$input,"tools":[]}""")
            assertEquals(0, attempts)
            assertTrue(stream.contains("Saved and verified dog.html"))
            assertFalse(stream.contains("response.failed"))
        }
    }
    @Test fun inferenceIsConstrainedToTheOfferedTools() {
        var seen: String? = null
        LocalCodexServer(infer = { error("Expected streaming inference") }, modelError = "Invalid",
            streamInfer = { _, grammar, _ -> seen = grammar; """{"text":"Listing","tool_calls":[{"name":"exec_command","arguments":{"cmd":"ls"}}]}""" }).use { server ->
            post("${server.baseUrl}/responses", """{"input":"list files","tools":[{"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}]}""")
            assertTrue(seen.orEmpty().contains("\"\\\"exec_command\\\"\""))
            assertTrue(seen.orEmpty().contains("\"\\\"cmd\\\"\" ws \":\" ws string"))
        }
    }
    @Test fun proseAfterASuccessfulWriteFinishesThroughTheReadBack() {
        val encoded = java.util.Base64.getEncoder().encodeToString("<h1>Dogs</h1>".toByteArray())
        var attempts = 0
        LocalCodexServer(infer = { attempts++; "The previous write_file call succeeded, so the dog site is done." },
            modelError = "Invalid").use { server ->
            val request = org.json.JSONObject().put("input", org.json.JSONArray()
                .put(org.json.JSONObject().put("type", "message").put("role", "user").put("content", "make a dog site"))
                .put(org.json.JSONObject().put("type", "function_call").put("name", "exec_command").put("call_id", "write")
                    .put("arguments", org.json.JSONObject().put("cmd", "printf '%s' '$encoded' | base64 -d > 'dog.html'").toString()))
                .put(org.json.JSONObject().put("type", "function_call_output").put("call_id", "write")
                    .put("output", "Process exited with code 0\nOutput:\n")))
                .put("tools", org.json.JSONArray().put(org.json.JSONObject().put("type", "function").put("name", "exec_command")
                    .put("parameters", org.json.JSONObject().put("required", org.json.JSONArray().put("cmd")))))
            val (_, stream) = post("${server.baseUrl}/responses", request.toString())
            assertEquals(1, attempts)
            assertFalse(stream.contains("response.failed"))
            val call = stream.lineSequence().filter { it.startsWith("data: ") }.map { org.json.JSONObject(it.removePrefix("data: ")) }
                .mapNotNull { it.optJSONObject("item") }.first { it.optString("type") == "function_call" }
            assertEquals("cat \"dog.html\"", org.json.JSONObject(call.getString("arguments")).getString("cmd"))
        }
    }
    @Test fun reportsProgressBeforeInferenceFinishesAndAcrossAgentSteps() {
        val updates = mutableListOf<Pair<Int, String>>()
        LocalCodexServer(infer = { error("Expected streaming inference") }, modelError = "Invalid response",
            streamInfer = { _, _, emit ->
                emit("<think>Reading files")
                assertTrue(updates.any { it.second.contains("Reading files") })
                """{"text":"Ready","tool_calls":[]}"""
            }, onProgress = { _, step, _, raw -> updates.add(step to raw) }).use { server ->
            repeat(2) { post("${server.baseUrl}/responses", """{"input":"hello","tools":[]}""") }
            assertTrue(updates.any { it.first == 1 && it.second.isNotBlank() })
            assertTrue(updates.any { it.first == 2 && it.second.isNotBlank() })
        }
    }
    @Test fun malformedCompletionIsRetriedWithoutPublishingIt() {
        var attempts = 0
        LocalCodexServer(infer = { error("Expected streaming inference") }, modelError = "Invalid response",
            streamInfer = { prompt, _, emit ->
                attempts++
                if (attempts == 1) {
                    val broken = """{"text":"I finished updating everything!"]}"""
                    emit(broken)
                    broken
                } else {
                    assertTrue(prompt.contains("were NOT executed"))
                    assertTrue(prompt.contains("tool_calls"))
                    """{"text":"Reading the project","tool_calls":[{"name":"shell_command","arguments":{"command":"ls"}}]}"""
                }
            }).use { server ->
            val (_, stream) = post("${server.baseUrl}/responses",
                """{"input":"edit the project","tools":[{"type":"function","name":"shell_command","parameters":{"required":["command"]}}]}""")
            assertEquals(2, attempts)
            assertFalse(stream.contains("I finished updating everything"))
            assertFalse(stream.contains("response.failed"))
            assertTrue(stream.contains("response.completed"))
            val events = stream.lineSequence().filter { it.startsWith("data: ") }
                .map { org.json.JSONObject(it.removePrefix("data: ")) }.toList()
            assertEquals(1, events.count { it.optString("type") == "response.output_text.delta" })
            assertEquals(1, events.count { it.optString("type") == "response.output_item.added" &&
                it.optJSONObject("item")?.optString("type") == "message" })
            assertTrue(events.any { it.optJSONObject("item")?.optString("type") == "function_call" })
        }
    }

    @Test fun repeatedMalformedResponsesStopAfterThreeAttempts() {
        var attempts = 0
        LocalCodexServer(infer = { attempts++; """{"text":"fake completion"]}""" }, modelError = "Invalid response").use { server ->
            val (_, stream) = post("${server.baseUrl}/responses", """{"input":"edit","tools":[]}""")
            assertEquals(3, attempts)
            assertTrue(stream.contains("response.failed"))
            assertFalse(stream.contains("response.output_text.delta"))
            assertFalse(stream.contains("response.completed"))
        }
    }

    private fun post(url: String, body: String): Pair<Int, String> {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.requestMethod = "POST"; conn.doOutput = true
        conn.readTimeout = 5000; conn.connectTimeout = 5000
        val bytes = body.toByteArray()
        conn.setFixedLengthStreamingMode(bytes.size)
        return try {
            conn.outputStream.use { it.write(bytes) }
            val status = conn.responseCode
            val text = (if (status < 400) conn.inputStream else conn.errorStream)?.bufferedReader()?.use { it.readText() }.orEmpty()
            status to text
        } finally { conn.disconnect() }
    }
    @Test fun localInferenceProducesCompleteResponsesStream() {
        var seen = ""
        LocalCodexServer(infer = { seen = it; """{"text":"Passed","tool_calls":[]}""" }, modelError = "Invalid local response").use {
            val (status, stream) = post("${it.baseUrl}/responses", """{"input":"run tests","tools":[]}""")
            assertEquals(200, status)
            assertTrue(seen.contains("run tests"))
            assertTrue(stream.contains("response.output_text.delta"))
            assertTrue(stream.contains("Passed"))
            assertTrue(stream.contains("response.completed"))
        }
    }
    @Test fun malformedToolOutputFailsWithoutReportingSuccess() {
        LocalCodexServer(infer = { "pretend edits" }, modelError = "Invalid local response").use {
            // Prose is a final answer, but an edit request still needs evidence that something changed.
            val (_, stream) = post("${it.baseUrl}/responses", """{"input":[{"type":"message","role":"user","content":"edit index.html"}],"tools":[]}""")
            assertTrue(stream.contains("local_model_error"))
            assertFalse(stream.contains("response.completed"))
        }
    }
    @Test fun rejectsRequestsWithoutLocalCapability() {
        var invoked = false
        LocalCodexServer(infer = { invoked = true; "" }, modelError = "Invalid local response").use {
            val wrong = it.baseUrl.replace(Regex("/[a-f0-9-]+/v1$"), "/incorrect/v1")
            assertEquals(404, post("$wrong/responses", "{}").first)
            assertFalse(invoked)
        }
    }
}
