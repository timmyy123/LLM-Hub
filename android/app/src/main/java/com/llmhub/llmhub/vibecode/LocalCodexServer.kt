package com.llmhub.llmhub.vibecode

import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.Closeable
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.UUID

/** Loopback only, with an unguessable per-process capability in the URL. No cloud forwarding. */
internal class LocalCodexServer(
    private val infer: suspend (String) -> String,
    private val modelError: String,
    /** Streams a reply to the prompt; the second argument is a GBNF grammar the reply must follow, if any. */
    private val streamInfer: (suspend (String, String?, (String) -> Unit) -> String)? = null,
    private val onRequest: (JSONObject) -> Unit = {},
    private val onProgress: (String, Int, Int, String) -> Unit = { _, _, _, _ -> },
    private val verifiedFileSummary: (String) -> String = { "Saved $it and verified its contents." }
) : Closeable {
    // A dropped connection or a bad request ends that request, never the app.
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO + CoroutineExceptionHandler { _, e ->
        runCatching { android.util.Log.w("LocalCodexServer", "Request ended", e) }
    })
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
        val http = LoopbackRequest.read(socket)
        val out = http.out
        if (http.method == "GET" && http.path == "/$token/v1/models") {
            val modelObj = JSONObject()
                .put("id", "llmhub-local")
                .put("slug", "llmhub-local")
                .put("name", "LLM Hub local")
                .put("supports_parallel_tool_calls", false)
            http.json("200 OK", JSONObject().put("models", org.json.JSONArray().put(modelObj)))
            return
        }
        if (http.method != "POST" || http.path != "/$token/v1/responses") {
            http.json("404 Not Found", JSONObject()); return
        }
        val body = http.body() ?: return
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
            // A failed write means Codex is gone: cancel the request, inference included, instead of failing.
            val requestScope = this
            val heartbeat = launch {
                while (isActive) {
                    delay(5_000)
                    if (runCatching { synchronized(lock) { out.write(": keepalive\n\n".toByteArray()); out.flush() } }.isFailure) requestScope.cancel()
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
                // A successful read-back that this server requested ends the turn; asking the model again for a
                // closing line invites it to doubt or redo the edit it just made.
                val grammar = CodexGrammar.build(request)
                var output: org.json.JSONArray? = CodexProgressGuard.latestResult(request)?.takeIf { (call, result) ->
                    call.optString("call_id").startsWith(READ_BACK_CALL) && result.contains("Process exited with code 0")
                }?.let { (call, _) ->
                    val summary = JSONObject().put("text", verifiedFileSummary(CodexProgressGuard.inspectionKey(call).orEmpty()
                        .removePrefix("cat "))).put("tool_calls", org.json.JSONArray()).toString()
                    onProgress(messageId, step, 1, summary)
                    CodexResponses.output(summary, request)
                }
                for (attempt in 0..2) {
                    if (output != null) break
                    val preview = StringBuilder()
                    var lastUpdate = 0L
                    onProgress(messageId, step, attempt + 1, "")
                    // Publish assistant text only after validation; rejected attempts stay out of chat.
                    val raw = try {
                        streamInfer?.invoke(inferencePrompt, grammar) { chunk ->
                            preview.append(chunk)
                            val now = System.nanoTime()
                            if (now - lastUpdate >= 200_000_000) {
                                onProgress(messageId, step, attempt + 1, preview.toString())
                                lastUpdate = now
                            }
                            // Backends without a grammar keep generating after the reply; stop at its last brace.
                            if (preview.length > MAX_REPLY_CHARS || preview.length % 256 < chunk.length && CodexResponses.degenerate(preview.toString()) ||
                        ('}' in chunk || ']' in chunk) && CodexResponses.replyComplete(preview.toString())) throw ReplyComplete()
                        }
                    } catch (_: ReplyComplete) { preview.toString() }
                        ?: infer(inferencePrompt)
                    onProgress(messageId, step, attempt + 1, raw)
                    try {
                        var nativeSummary: String? = null
                        var candidate = try { CodexResponses.output(raw, request) }
                            catch (error: org.json.JSONException) {
                                // Plain prose with no JSON or tool markup is a final answer (LiteRT-LM has no grammar to
                                // keep Gemma to JSON); the progress guard still demands evidence before accepting it.
                                val completion = CodexResponses.nativeCompletion(raw)
                                    ?: CodexResponses.stripThinking(raw).takeIf { it.isNotBlank() && '{' !in it && "tool_call" !in it }
                                    ?: throw error
                                nativeSummary = completion
                                CodexResponses.output(completion, request, allowPlainText = true)
                            }
                        CodexProgressGuard.verifiedDuplicateRead(candidate, request)?.let { path ->
                            nativeSummary = verifiedFileSummary(path)
                            candidate = CodexResponses.output(JSONObject().put("text", nativeSummary)
                                .put("tool_calls", org.json.JSONArray()).toString(), request)
                        }
                        try {
                            CodexProgressGuard.validate(candidate, request)
                        } catch (e: UnverifiedCompletionException) {
                            // The edit itself succeeded; read the file back for the model instead of failing the turn.
                            val path = CodexProgressGuard.unverifiedEdit(request) ?: throw e
                            candidate = CodexResponses.output(JSONObject().put("text", "").put("tool_calls", org.json.JSONArray()
                                .put(JSONObject().put("name", "read_file").put("arguments", JSONObject().put("path", path)))).toString(), request)
                            for (i in 0 until candidate.length()) candidate.getJSONObject(i).takeIf { it.has("call_id") }
                                ?.put("call_id", READ_BACK_CALL + UUID.randomUUID())
                            CodexProgressGuard.validate(candidate, request)
                            nativeSummary = null
                        }
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

    private companion object {
        const val READ_BACK_CALL = "call_llmhub_readback_"
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
