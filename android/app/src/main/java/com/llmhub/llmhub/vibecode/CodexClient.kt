package com.llmhub.llmhub.vibecode

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.withTimeout
import okhttp3.*
import org.json.JSONObject
import java.io.Closeable
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

internal class CodexClient : Closeable {
    private val http = OkHttpClient.Builder().readTimeout(0, TimeUnit.MILLISECONDS).build()
    private var socket: WebSocket? = null
    private val nextId = AtomicInteger()
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<JSONObject>>()
    val events = Channel<JSONObject>(Channel.UNLIMITED)

    suspend fun connect(port: Int, token: String) {
        val opened = CompletableDeferred<Unit>()
        socket = http.newWebSocket(Request.Builder().url("ws://127.0.0.1:$port")
            .header("Authorization", "Bearer $token").build(), object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) { opened.complete(Unit) }
            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val msg = JSONObject(text)
                    if (msg.has("id") && !msg.has("method")) {
                        pending.remove(msg.getInt("id"))?.let {
                            if (msg.has("error")) it.completeExceptionally(IOException(msg.getJSONObject("error").optString("message")))
                            else it.complete(msg.optJSONObject("result") ?: JSONObject())
                        }
                    } else events.trySend(msg)
                } catch (e: Exception) { fail(e) }
            }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                opened.completeExceptionally(t); fail(t)
            }
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) { fail(IOException(reason)) }
        })
        withTimeout(10_000) { opened.await() }
        request("initialize", JSONObject().put("clientInfo", JSONObject().put("name", "llmhub_android")
            .put("title", "LLM Hub").put("version", "1.0")))
        send(JSONObject().put("method", "initialized"))
    }

    suspend fun request(method: String, params: JSONObject = JSONObject()): JSONObject {
        val id = nextId.incrementAndGet()
        val result = CompletableDeferred<JSONObject>()
        pending[id] = result
        try {
            send(JSONObject().put("id", id).put("method", method).put("params", params))
            return withTimeout(60_000) { result.await() }
        } finally { pending.remove(id) }
    }

    fun respond(id: Any, result: JSONObject) = send(JSONObject().put("id", id).put("result", result))
    fun reject(id: Any) = send(JSONObject().put("id", id).put("error",
        JSONObject().put("code", -32601).put("message", "Unsupported client request")))
    private fun send(msg: JSONObject) { check(socket?.send(msg.toString()) == true) { "Codex disconnected" } }
    private fun fail(t: Throwable) {
        pending.values.forEach { it.completeExceptionally(t) }; pending.clear(); events.close(t)
    }
    override fun close() {
        socket?.cancel(); fail(IOException("Codex connection closed"))
        http.dispatcher.executorService.shutdown(); http.connectionPool.evictAll()
    }
}
