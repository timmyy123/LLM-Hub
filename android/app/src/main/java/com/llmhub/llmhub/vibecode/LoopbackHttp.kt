package com.llmhub.llmhub.vibecode

import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.OutputStream
import java.net.Socket

/** One HTTP/1.1 request on a loopback model server connection: headers parsed, body read on demand. */
internal class LoopbackRequest(
    val method: String,
    /** The request path without its query string. */
    val path: String,
    val headers: Map<String, String>,
    private val input: BufferedInputStream,
    val out: OutputStream
) {
    /** The body, or null after answering 413 when it has no usable Content-Length (chunked bodies are refused). */
    fun body(): ByteArray? {
        if (headers["expect"]?.lowercase() == "100-continue") {
            out.write("HTTP/1.1 100 Continue\r\n\r\n".toByteArray()); out.flush()
        }
        val length = headers["content-length"]?.toIntOrNull()
        if (length == null || length !in 1..8_388_608 || headers.containsKey("transfer-encoding")) {
            json("413 Payload Too Large", JSONObject()); return null
        }
        val body = ByteArray(length)
        var read = 0
        while (read < length) {
            val n = input.read(body, read, length - read)
            require(n > 0); read += n
        }
        return body
    }

    fun json(status: String, value: JSONObject) {
        val bytes = value.toString().toByteArray()
        out.write("HTTP/1.1 $status\r\nContent-Type: application/json\r\nContent-Length: ${bytes.size}\r\nConnection: close\r\n\r\n".toByteArray())
        out.write(bytes); out.flush()
    }

    companion object {
        fun read(socket: Socket): LoopbackRequest {
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
            return LoopbackRequest(first.getOrElse(0) { "" }, first.getOrElse(1) { "" }.substringBefore('?'),
                headers, input, socket.getOutputStream())
        }
    }
}
