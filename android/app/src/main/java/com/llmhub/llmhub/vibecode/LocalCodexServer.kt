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
    private val onRequest: (JSONObject) -> Unit = {}
) : Closeable {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val listener = ServerSocket(0, 4, InetAddress.getByName("127.0.0.1"))
    private val token = UUID.randomUUID().toString()
    val baseUrl = "http://127.0.0.1:${listener.localPort}/$token/v1"
    private val sockets = java.util.concurrent.ConcurrentHashMap.newKeySet<Socket>()

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
                var liveText = ""
                var messageAdded = false
                val partial = StringBuilder()
                fun onChunk(chunk: String) {
                    partial.append(chunk)
                    val preview = CodexResponses.partialText(partial.toString())
                    if (preview.length <= liveText.length || !preview.startsWith(liveText)) return
                    if (!messageAdded) {
                        event(CodexResponses.event("response.output_item.added", "output_index" to 0,
                            "item" to JSONObject().put("id", messageId).put("type", "message").put("role", "assistant")
                                .put("status", "in_progress").put("content", org.json.JSONArray())))
                        messageAdded = true
                    }
                    event(CodexResponses.event("response.output_text.delta", "item_id" to messageId,
                        "output_index" to 0, "content_index" to 0, "delta" to preview.substring(liveText.length)))
                    liveText = preview
                }
                val prompt = CodexResponses.prompt(request)
                val raw = streamInfer?.invoke(prompt, ::onChunk) ?: infer(prompt)
                val output = CodexResponses.output(raw, request)
                for (i in 0 until output.length()) {
                    val item = output.getJSONObject(i)
                    if (item.optString("type") == "message") {
                        item.put("id", messageId)
                        if (!messageAdded) event(CodexResponses.event("response.output_item.added", "output_index" to i, "item" to item))
                        val text = item.getJSONArray("content").getJSONObject(0).getString("text")
                        val delta = if (text.startsWith(liveText)) text.substring(liveText.length) else text
                        if (delta.isNotEmpty()) event(CodexResponses.event("response.output_text.delta", "item_id" to messageId,
                            "output_index" to i, "content_index" to 0, "delta" to delta))
                    } else event(CodexResponses.event("response.output_item.added", "output_index" to i, "item" to item))
                    event(CodexResponses.event("response.output_item.done", "output_index" to i, "item" to item))
                }
                event(CodexResponses.event("response.completed", "response" to JSONObject().put("id", id)
                    .put("status", "completed").put("output", output)))
            } catch (e: CancellationException) { throw e }
            catch (_: Exception) {
                event(CodexResponses.event("response.failed", "response" to JSONObject().put("id", id).put("status", "failed")
                    .put("error", JSONObject().put("type", "invalid_request_error")
                        .put("code", "local_model_error").put("message", modelError))))
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
