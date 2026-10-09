package com.llmhub.llmhub.vibecode

import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.Closeable
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.UUID

/** Loopback only, with an unguessable per-process capability in the URL. No cloud forwarding. */
internal class LocalCodexServer(
    private val infer: suspend (String) -> String,
    private val modelError: String,
    private val streamInfer: (suspend (String, (String) -> Unit) -> String)? = null,
    private val onRequest: (JSONObject) -> Unit = {},
    private val onProgress: (String, Int, Int, String) -> Unit = { _, _, _, _ -> }
) : Closeable {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val listener = ServerSocket(0, 4, InetAddress.getByName("127.0.0.1"))
    private val token = UUID.randomUUID().toString()
    val baseUrl = "http://127.0.0.1:${listener.localPort}/$token/v1"
    private val sockets = java.util.concurrent.ConcurrentHashMap.newKeySet<Socket>()
    private val steps = java.util.concurrent.atomic.AtomicInteger()

    init {
        scope.launch {
            while (isActive) {
                val socket = try { listener.accept() } catch (_: Exception) { break }
                sockets.add(socket)
                launch { try { socket.use { handle(it) } } finally { sockets.remove(socket) } }
            }
        }
    }

    private suspend fun handle(socket: Socket) {
        socket.soTimeout = 30_000
        val input = BufferedInputStream(socket.getInputStream())
        fun line(): String {
            val bytes = java.io.ByteArrayOutputStream()
            while (true) {
                val c = input.read()
                require(c >= 0) { "Incomplete HTTP header" }
                if (c == 10) break
                require(bytes.size() < 8192)
                if (c != 13) bytes.write(c)
            }
            return bytes.toString("UTF-8")
        }
        val first = line().split(' ')
        val headers = mutableMapOf<String, String>()
        var headerBytes = 0
        while (true) {
            val header = line()
            if (header.isEmpty()) break
            headerBytes += header.length
            require(headerBytes < 32768)
            headers[header.substringBefore(':').lowercase()] = header.substringAfter(':').trim()
        }
        val out = socket.getOutputStream()
        fun json(status: String, value: JSONObject) {
            val bytes = value.toString().toByteArray()
            out.write("HTTP/1.1 $status\r\nContent-Type: application/json\r\nContent-Length: ${bytes.size}\r\nConnection: close\r\n\r\n".toByteArray())
            out.write(bytes); out.flush()
        }
        if (first.getOrNull(0) == "GET" && first.getOrNull(1)?.substringBefore('?') == "/$token/v1/models") {
            val modelObj = JSONObject()
                .put("id", "llmhub-local")
                .put("slug", "llmhub-local")
                .put("name", "LLM Hub local")
                .put("supports_parallel_tool_calls", false)
            json("200 OK", JSONObject().put("models", org.json.JSONArray().put(modelObj)))
            return
        }
        if (first.getOrNull(0) != "POST" || first.getOrNull(1)?.substringBefore('?') != "/$token/v1/responses") {
            json("404 Not Found", JSONObject()); return
        }
        if (headers["expect"]?.lowercase() == "100-continue") {
            out.write("HTTP/1.1 100 Continue\r\n\r\n".toByteArray()); out.flush()
        }
        val length = headers["content-length"]?.toIntOrNull()
        if (length == null || length !in 1..8_388_608 || headers.containsKey("transfer-encoding")) {
            json("413 Payload Too Large", JSONObject()); return
        }
        val body = ByteArray(length)
        var read = 0
        while (read < length) {
            val n = input.read(body, read, length - read)
            require(n > 0); read += n
        }
        val request = JSONObject(String(body, Charsets.UTF_8))
        onRequest(request)
        val id = "resp_${UUID.randomUUID()}"
        // Send headers immediately and heartbeat while the local model decodes structured output.
        out.write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n".toByteArray())
        val lock = Any()
        fun event(value: JSONObject) = synchronized(lock) {
            out.write("event: ${value.getString("type")}\ndata: $value\n\n".toByteArray()); out.flush()
        }
        event(CodexResponses.event("response.created", "response" to JSONObject().put("id", id)))
        coroutineScope {
            val heartbeat = launch {
                while (isActive) {
                    delay(5_000)
                    synchronized(lock) { out.write(": keepalive\n\n".toByteArray()); out.flush() }
                }
            }
            try {
                val messageId = "msg_${UUID.randomUUID()}"
                val step = steps.incrementAndGet()
                val prompt = CodexResponses.prompt(request)
                runCatching { android.util.Log.d("LocalCodexServer", "Codex prompt length: ${prompt.length}") }
                var inferencePrompt = prompt
                val rejectionReasons = mutableListOf<String>()
                fun recover(error: Exception): String {
                    error.message?.let { if (it !in rejectionReasons) rejectionReasons.add(it) }
                    return CodexResponses.recoveryPrompt(request,
                        IllegalArgumentException(rejectionReasons.joinToString("\nPrevious rejection: ")))
                }
                var output: org.json.JSONArray? = null
                for (attempt in 0..2) {
                    val preview = StringBuilder()
                    var lastUpdate = 0L
                    onProgress(messageId, step, attempt + 1, "")
                    // Publish assistant text only after validation; rejected attempts stay out of chat.
                    val raw = streamInfer?.invoke(inferencePrompt) { chunk ->
                        preview.append(chunk)
                        val now = System.nanoTime()
                        if (now - lastUpdate >= 200_000_000) {
                            onProgress(messageId, step, attempt + 1, preview.toString())
                            lastUpdate = now
                        }
                    }
                        ?: infer(inferencePrompt)
                    onProgress(messageId, step, attempt + 1, raw)
                    try {
                        var nativeSummary: String? = null
                        val candidate = try { CodexResponses.output(raw, request) }
                            catch (error: org.json.JSONException) {
                                val completion = CodexResponses.nativeCompletion(raw) ?: throw error
                                nativeSummary = completion
                                CodexResponses.output(completion, request, allowPlainText = true)
                            }
                        CodexProgressGuard.validate(candidate, request)
                        nativeSummary?.let { summary ->
                            // Replace the provisional thinking preview before publishing this same final text.
                            onProgress(messageId, step, attempt + 1,
                                JSONObject().put("text", summary).put("tool_calls", org.json.JSONArray()).toString())
                        }
                        output = candidate
                        break
                    } catch (e: IllegalArgumentException) {
                        if (attempt == 2) throw e
                        inferencePrompt = recover(e)
                    } catch (e: org.json.JSONException) {
                        if (attempt == 2) throw e
                        inferencePrompt = recover(e)
                    }
                }
                val validated = checkNotNull(output)
                for (i in 0 until validated.length()) {
                    val item = validated.getJSONObject(i)
                    if (item.optString("type") == "message") {
                        item.put("id", messageId)
                        event(CodexResponses.event("response.output_item.added", "output_index" to i, "item" to item))
                        val text = item.getJSONArray("content").getJSONObject(0).getString("text")
                        if (text.isNotEmpty()) event(CodexResponses.event("response.output_text.delta", "item_id" to messageId,
                            "output_index" to i, "content_index" to 0, "delta" to text))
                    } else event(CodexResponses.event("response.output_item.added", "output_index" to i, "item" to item))
                    event(CodexResponses.event("response.output_item.done", "output_index" to i, "item" to item))
                }
                event(CodexResponses.event("response.completed", "response" to JSONObject().put("id", id)
                    .put("status", "completed").put("output", validated)))
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                runCatching { android.util.Log.e("LocalCodexServer", "Error generating or parsing response", e) }
                event(CodexResponses.event("response.failed", "response" to JSONObject().put("id", id).put("status", "failed")
                    .put("error", JSONObject().put("type", "invalid_request_error")
                        .put("code", "local_model_error").put("message", e.message ?: modelError))))
            } finally { heartbeat.cancel() }
        }
    }

    suspend fun shutdown() {
        close()
        scope.coroutineContext[Job]?.join()
    }

    override fun close() {
        listener.close()
        sockets.forEach { runCatching { it.close() } }
        scope.cancel()
    }
}
