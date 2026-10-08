package com.llmhub.llmhub.agent

import java.io.Closeable
import java.io.InputStreamReader
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.UUID

/** Authenticated local byte stream: stdout and stderr arrive before RUN_COMMAND completes. */
internal class TermuxOutputStream : Closeable {
    private val listener = ServerSocket(0, 4, InetAddress.getByName("127.0.0.1"))
    private val token = UUID.randomUUID().toString()
    @Volatile private var peer: Socket? = null
    @Volatile var processGroup: Int? = null
        private set

    fun commandScript(command: String): String {
        fun quote(value: String) = "'" + value.replace("'", "'\"'\"'") + "'"
        val child = """
            printf '%s\n' "${'$'}${'$'}" >&3
            exec 1>&3 2>&3
            exec 3>&-
            $command
        """.trimIndent()
        return """
            export PATH=/data/data/com.termux/files/usr/bin:${'$'}PATH
            exec 3<>/dev/tcp/127.0.0.1/${listener.localPort} || exit 1
            printf '%s\n' ${quote(token)} >&3
            set -m
            bash -c ${quote(child)} < /dev/null
            exit ${'$'}?
        """.trimIndent()
    }

    /** Blocking IO, called on Dispatchers.IO. Closing the stream unblocks reads and accept. */
    fun collect(onOutput: (String) -> Unit) {
        while (!listener.isClosed) {
            val socket = listener.accept()
            peer = socket
            socket.use {
                socket.soTimeout = 5000
                val input = socket.getInputStream()
                fun header(): String {
                    val bytes = java.io.ByteArrayOutputStream()
                    while (true) {
                        val c = input.read()
                        require(c >= 0 && bytes.size() < 128) { "Invalid terminal stream header" }
                        if (c == 10) return bytes.toString("UTF-8")
                        bytes.write(c)
                    }
                }
                if (runCatching { header() }.getOrNull() != token) return@use
                processGroup = header().toInt().also { require(it > 1) }
                socket.soTimeout = 0
                val reader = InputStreamReader(input, Charsets.UTF_8)
                val chars = CharArray(4096)
                while (true) {
                    val n = reader.read(chars)
                    if (n < 0) return
                    onOutput(String(chars, 0, n))
                }
            }
            peer = null
        }
    }

    override fun close() {
        listener.close()
        runCatching { peer?.close() }
    }
}

/** Keep terminal updates bounded and remove terminal-control codes from Compose text. */
internal class TerminalOutputBuffer(private val limit: Int = 100_000) {
    private val text = StringBuilder()
    @Synchronized fun append(chunk: String): String {
        text.append(chunk)
        if (text.length > limit) text.delete(0, text.length - limit)
        return text.toString()
            .replace(Regex("\u001B\\[[0-?]*[ -/]*[@-~]"), "")
            .replace(Regex("\u001B\\][^\u0007]*(?:\u0007|\u001B\\\\)"), "")
            .replace("\r\n", "\n").replace('\r', '\n')
    }
}
