package com.llmhub.llmhub.vibecode

import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.Closeable
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.UUID

/**
 * The Anthropic Messages API on loopback for Claude Code, answered by the local model. Like [LocalCodexServer]:
 * an unguessable capability in the URL, no cloud forwarding.
 */
internal class LocalClaudeServer(
    /** Streams a reply to the prompt; the second argument is a GBNF grammar the reply must follow, if any. */
    private val infer: suspend (String, String?, (String) -> Unit) -> String,
    private val onProgress: (String, Int, Int, String) -> Unit = { _, _, _, _ -> }
) : Closeable {
    // A dropped connection or a bad request ends that request, never the app.
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO + CoroutineExceptionHandler { _, e ->
        runCatching { android.util.Log.w("LocalClaudeServer", "Request ended", e) }
    })
    private val listener = ServerSocket(0, 4, InetAddress.getByName("127.0.0.1"))
    private val token = UUID.randomUUID().toString()
    /** For ANTHROPIC_BASE_URL; Claude Code appends /v1/messages. */
    val baseUrl = "http://127.0.0.1:${listener.localPort}/$token"
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
        if (http.method != "POST" || !http.path.startsWith("/$token/v1/messages")) {
            http.json("404 Not Found", JSONObject()); return
        }
        val request = JSONObject(String(http.body() ?: return, Charsets.UTF_8))
        val prompt = ClaudeMessages.prompt(request)
        if (http.path == "/$token/v1/messages/count_tokens") {
            http.json("200 OK", JSONObject().put("input_tokens", prompt.length / 4)); return
        }
        if (http.path != "/$token/v1/messages") { http.json("404 Not Found", JSONObject()); return }
        val id = "msg_${UUID.randomUUID().toString().replace("-", "")}"
        val model = request.optString("model", "llmhub-local")
        val stream = request.optBoolean("stream")
        val out = http.out
        val lock = Any()
        fun event(value: JSONObject) = synchronized(lock) {
            out.write("event: ${value.getString("type")}\ndata: $value\n\n".toByteArray()); out.flush()
        }
        if (stream) out.write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n".toByteArray())
        coroutineScope {
            // Anthropic's own keep-alive while the local model decodes.
            // A failed write means the client is gone: cancel the request, inference included, instead of failing.
            val requestScope = this
            val heartbeat = if (stream) launch { while (isActive) { delay(5_000); if (runCatching { event(JSONObject().put("type", "ping")) }.isFailure) requestScope.cancel() } } else null
            try {
                val content = if ((request.optJSONArray("tools")?.length() ?: 0) == 0) {
                    // Claude Code's side requests (session titles, quota probes) carry no tools; they need no model run.
                    org.json.JSONArray().put(JSONObject().put("type", "text").put("text", "OK"))
                } else generate(request, prompt)
                val message = ClaudeMessages.message(id, model, content, prompt.length / 4,
                    maxOf(1, content.toString().length / 4))
                if (stream) ClaudeMessages.events(message).forEach(::event) else http.json("200 OK", message)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                runCatching { android.util.Log.e("LocalClaudeServer", "Error generating or parsing response", e) }
                val error = JSONObject().put("type", "error").put("error", JSONObject()
                    .put("type", "invalid_request_error").put("message", e.message ?: "Local model error"))
                if (stream) event(error) else http.json("400 Bad Request", error)
            } finally { heartbeat?.cancel() }
        }
    }

    private suspend fun generate(request: JSONObject, prompt: String): org.json.JSONArray {
        val grammar = CodexGrammar.forTools(ClaudeMessages.offeredTools(request))
        val step = steps.incrementAndGet()
        val progressId = "claude-step-$step"
        var inferencePrompt = prompt
        var attemptGrammar = grammar
        for (attempt in 1..3) {
            onProgress(progressId, step, attempt, "")
            val preview = StringBuilder()
            var lastUpdate = 0L
            val raw = try {
                infer(inferencePrompt, attemptGrammar) { chunk ->
                    preview.append(chunk)
                    val now = System.nanoTime()
                    if (now - lastUpdate >= 200_000_000) { onProgress(progressId, step, attempt, preview.toString()); lastUpdate = now }
                    if (preview.length > MAX_REPLY_CHARS || preview.length % 256 < chunk.length && CodexResponses.degenerate(preview.toString()) ||
                        ('}' in chunk || ']' in chunk) && CodexResponses.replyComplete(preview.toString())) throw ReplyComplete()
                }
            } catch (_: ReplyComplete) { preview.toString() }
            onProgress(progressId, step, attempt, raw)
            try {
                val content = ClaudeMessages.content(raw, request)
                if (ClaudeMessages.stopReason(content) != "tool_use") {
                    // Small models claim work ("I've created dog.html") without calling a single tool. Until some tool
                    // has run for this request, retries must call one: the grammar makes that certain on llama.cpp.
                    if (!ClaudeMessages.toolUsedSinceRequest(request)) {
                        attemptGrammar = CodexGrammar.forTools(ClaudeMessages.offeredTools(request), requireCall = true)
                        throw IllegalArgumentException("No tool has run for this request, so nothing was read or changed. " +
                            "Call the tool the request needs now, for example Write or Edit for a change or Read to look at a file.")
                    }
                    // Tools ran but nothing changed ("I will add a button" after a Read): ask once, then believe it.
                    require(attempt > 1 || ClaudeMessages.changedSinceRequest(request)) {
                        "Nothing has been changed for this request yet. If it needs a change, call Write or Edit now. " +
                            "If it needs none, reply again with tool_calls: [] and say briefly why."
                    }
                }
                return content
            } catch (e: Exception) {
                if (e !is IllegalArgumentException && e !is org.json.JSONException || attempt == 3) throw e
                inferencePrompt = "$prompt\n\nYour previous reply was rejected: ${e.message?.take(400)}\n" +
                    "Return complete valid JSON with text and tool_calls, using only the tools and argument names above."
            }
        }
        error("unreachable")
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

/** Thrown from the streaming callback to stop generation once a whole reply has arrived, or it ran too long. */
internal class ReplyComplete : RuntimeException()

/** Generous for one tool call (a whole small file); backends without a token limit (LiteRT-LM) otherwise run on. */
internal const val MAX_REPLY_CHARS = 16_000
