package com.llmhub.llmhub.vibecode

import java.net.HttpURLConnection
import java.net.URL
import org.junit.Assert.*
import org.junit.Test

class LocalCodexServerTest {
    @Test fun assistantTextArrivesWhileLocalModelIsStillGenerating() {
        val release = java.util.concurrent.CountDownLatch(1)
        LocalCodexServer(infer = { error("Streaming inference must be used") }, modelError = "Invalid response",
            streamInfer = { _, emit ->
                emit("""{"text":"Early""")
                check(release.await(5, java.util.concurrent.TimeUnit.SECONDS))
                """{"text":"Early finished","tool_calls":[]}"""
            }).use { server ->
            val connection = URL("${server.baseUrl}/responses").openConnection() as HttpURLConnection
            connection.requestMethod = "POST"; connection.doOutput = true; connection.readTimeout = 5000
            val body = """{"input":"hello","tools":[]}""".toByteArray()
            connection.setFixedLengthStreamingMode(body.size)
            try {
                connection.outputStream.use { it.write(body) }
                val reader = connection.inputStream.bufferedReader()
                var line: String
                do { line = reader.readLine() ?: error("Stream closed before text") }
                while (!line.contains("\"delta\":\"Early\""))
                release.countDown()
                val rest = reader.readText()
                assertTrue(rest.contains("response.completed"))
                assertTrue(rest.contains(" finished"))
            } finally { release.countDown(); connection.disconnect() }
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
            val (_, stream) = post("${it.baseUrl}/responses", """{"input":"edit","tools":[]}""")
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
