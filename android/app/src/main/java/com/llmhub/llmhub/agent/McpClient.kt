package com.llmhub.llmhub.agent

import android.content.Context
import com.llmhub.llmhub.BuildConfig
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

data class McpTool(
    val name: String,
    val description: String,
    val inputSchema: JSONObject
) {
    val exposedName: String = "mcp_" + name.replace(Regex("[^A-Za-z0-9_]"), "_")

    fun promptLine(): String {
        val properties = inputSchema.optJSONObject("properties") ?: JSONObject()
        val required = inputSchema.optJSONArray("required")?.let { array ->
            (0 until array.length()).mapNotNull { array.optString(it) }.toSet()
        } ?: emptySet()
        val args = properties.keys().asSequence().map { key ->
            val type = properties.optJSONObject(key)?.optString("type", "value") ?: "value"
            "$key: $type${if (key in required) " (required)" else ""}"
        }.joinToString(", ")
        return "- $exposedName({$args}): ${description.ifBlank { name }.take(300)}"
    }
}

data class McpSettings(
    val enabled: Boolean,
    val transport: String,
    val url: String,
    val token: String,
    val command: String
)

/** Minimal MCP Streamable HTTP client for mobile Agent tool discovery and calls. */
class McpClient(private val context: Context) {
    companion object {
        private const val CURRENT_PROTOCOL = "2026-07-28"
        private const val LEGACY_PROTOCOL = "2025-06-18"
    }

    private val prefs = EncryptedSharedPreferences.create(
        context,
        "agent_mcp_secure",
        MasterKey.Builder(context).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
    )
    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(45, TimeUnit.SECONDS)
        .writeTimeout(15, TimeUnit.SECONDS)
        .build()
    private val ids = AtomicLong(1)
    private var sessionId: String? = null
    private var protocolVersion = CURRENT_PROTOCOL

    var tools: List<McpTool> = emptyList()
        private set

    private fun clientInfo(name: String = "LLM Hub Android") =
        JSONObject().put("name", name).put("version", BuildConfig.VERSION_NAME)

    fun loadSettings() = McpSettings(
        enabled = prefs.getBoolean("enabled", false),
        transport = prefs.getString("transport", "termux") ?: "termux",
        url = prefs.getString("url", "") ?: "",
        token = prefs.getString("token", "") ?: "",
        command = prefs.getString("command", "") ?: ""
    )

    fun saveSettings(settings: McpSettings) {
        prefs.edit().putBoolean("enabled", settings.enabled)
            .putString("transport", settings.transport)
            .putString("url", settings.url.trim())
            .putString("token", settings.token.trim())
            .putString("command", settings.command.trim()).apply()
        if (!settings.enabled) disconnect()
    }

    fun disconnect() {
        sessionId = null
        protocolVersion = CURRENT_PROTOCOL
        tools = emptyList()
    }

    suspend fun connectAndDiscover(settings: McpSettings = loadSettings()): List<McpTool> = withContext(Dispatchers.IO) {
        require(settings.enabled) { "MCP is disabled" }
        if (settings.transport == "termux") {
            require(settings.command.isNotBlank()) { "MCP Termux command is empty" }
            tools = parseTools(termuxRpc(settings.command, "tools/list", JSONObject()))
            return@withContext tools
        }
        validateUrl(settings.url)
        disconnect()

        // Current MCP is stateless. If a server still requires the 2025 lifecycle,
        // initialize it and retain its session ID before listing tools.
        val first = runCatching { rpc(settings, "tools/list", JSONObject(), CURRENT_PROTOCOL) }
        val result = if (first.isSuccess) {
            first.getOrThrow()
        } else {
            protocolVersion = LEGACY_PROTOCOL
            val init = rpc(
                settings,
                "initialize",
                JSONObject()
                    .put("protocolVersion", LEGACY_PROTOCOL)
                    .put("capabilities", JSONObject())
                    .put("clientInfo", clientInfo("LLM Hub")),
                LEGACY_PROTOCOL,
                includeProtocolHeader = false
            )
            protocolVersion = init.optString("protocolVersion", LEGACY_PROTOCOL)
            notify(settings, "notifications/initialized")
            rpc(settings, "tools/list", JSONObject(), protocolVersion)
        }
        tools = parseTools(result)
        tools
    }

    suspend fun callTool(name: String, arguments: JSONObject): String = withContext(Dispatchers.IO) {
        val settings = loadSettings()
        require(settings.enabled) { "MCP is disabled" }
        val result = if (settings.transport == "termux") {
            termuxRpc(settings.command, "tools/call", JSONObject().put("name", name).put("arguments", arguments))
        } else rpc(
            settings,
            "tools/call",
            JSONObject().put("name", name).put("arguments", arguments),
            protocolVersion,
            mcpName = name
        )
        val isError = result.optBoolean("isError", false)
        val output = buildList {
            val content = result.optJSONArray("content") ?: JSONArray()
            for (i in 0 until content.length()) {
                val item = content.optJSONObject(i) ?: continue
                when (item.optString("type")) {
                    "text" -> item.optString("text").takeIf { it.isNotBlank() }?.let(::add)
                    "resource_link" -> item.optString("uri").takeIf { it.isNotBlank() }?.let(::add)
                    else -> add(item.toString())
                }
            }
            result.opt("structuredContent")?.let { add(it.toString()) }
        }.joinToString("\n").ifBlank { result.toString() }
        if (isError) throw McpException(output)
        output.take(100_000)
    }

    /**
     * Termux RUN_COMMAND is one-shot rather than an attachable pipe. Each approved
     * request starts the configured stdio server, performs the MCP lifecycle and
     * one operation, then closes stdin. This keeps stdio support compatible with
     * Android's app sandbox without requiring LLM Hub to execute server binaries.
     */
    private suspend fun termuxRpc(command: String, method: String, params: JSONObject): JSONObject {
        val initId = ids.getAndIncrement()
        val requestId = ids.getAndIncrement()
        val messages = listOf(
            JSONObject().put("jsonrpc", "2.0").put("id", initId).put("method", "initialize").put(
                "params", JSONObject().put("protocolVersion", LEGACY_PROTOCOL).put("capabilities", JSONObject())
                    .put("clientInfo", clientInfo())
            ),
            JSONObject().put("jsonrpc", "2.0").put("method", "notifications/initialized"),
            JSONObject().put("jsonrpc", "2.0").put("id", requestId).put("method", method).put("params", params)
        ).joinToString("\n", postfix = "\n")
        val encoded = android.util.Base64.encodeToString(messages.toByteArray(), android.util.Base64.NO_WRAP)
        // RUN_COMMAND does not initialize Termux's full interactive environment.
        // In particular, npm launchers commonly use `#!/usr/bin/env node`, which
        // needs libtermux-exec to rewrite Linux shebang paths to Termux paths.
        val prefix = "/data/data/com.termux/files/usr"
        val home = "/data/data/com.termux/files/home"
        val shell = "export PREFIX='$prefix' HOME='$home' PATH='$prefix/bin:/system/bin' " +
            "LD_PRELOAD='$prefix/lib/libtermux-exec.so'; " +
            "printf '%s' '$encoded' | base64 -d | timeout 45s $command"
        val output = try {
            TermuxCommandBridge.run(context, shell, 50_000)
        } catch (_: kotlinx.coroutines.TimeoutCancellationException) {
            throw McpException("termux_command_timed_out")
        }
        val response = output.lineSequence().mapNotNull { line ->
            val start = line.indexOf('{')
            if (start < 0) null else runCatching { JSONObject(line.substring(start)) }.getOrNull()
        }.lastOrNull { it.opt("id")?.toString() == requestId.toString() }
            ?: throw McpException("MCP server returned no response: ${output.takeLast(500)}")
        response.optJSONObject("error")?.let { throw McpException(it.optString("message", "MCP request failed")) }
        return response.optJSONObject("result") ?: throw McpException("Invalid MCP response")
    }

    private fun validateUrl(raw: String) {
        val url = raw.trim().toHttpUrlOrNull() ?: throw McpException("Invalid MCP server URL")
        if (url.scheme != "https" && !isPrivateNetworkHost(url.host)) {
            throw McpException("MCP servers must use HTTPS")
        }
    }

    private fun isPrivateNetworkHost(host: String): Boolean {
        val normalized = host.lowercase()
        if (normalized in setOf("localhost", "::1", "10.0.2.2") || normalized.endsWith(".local")) {
            return true
        }
        if (normalized.startsWith("fe80:") || normalized.startsWith("fc") || normalized.startsWith("fd")) {
            return true
        }

        val octets = normalized.split('.').map { it.toIntOrNull() }
        if (octets.size != 4 || octets.any { it == null || it !in 0..255 }) return false
        val first = octets[0]!!
        val second = octets[1]!!
        return first == 10 ||
            first == 127 ||
            (first == 169 && second == 254) ||
            (first == 172 && second in 16..31) ||
            (first == 192 && second == 168)
    }

    private fun parseTools(result: JSONObject): List<McpTool> {
        val array = result.optJSONArray("tools") ?: JSONArray()
        return (0 until array.length()).mapNotNull { index ->
            val item = array.optJSONObject(index) ?: return@mapNotNull null
            val name = item.optString("name").takeIf { it.isNotBlank() } ?: return@mapNotNull null
            McpTool(name, item.optString("description"), item.optJSONObject("inputSchema") ?: JSONObject().put("type", "object"))
        }.take(64)
    }

    private fun rpc(
        settings: McpSettings,
        method: String,
        params: JSONObject,
        version: String,
        includeProtocolHeader: Boolean = true,
        mcpName: String? = null
    ): JSONObject {
        val id = ids.getAndIncrement()
        if (version == CURRENT_PROTOCOL) {
            params.put("_meta", JSONObject().put("io.modelcontextprotocol/clientInfo", clientInfo()))
        }
        val body = JSONObject().put("jsonrpc", "2.0").put("id", id).put("method", method).put("params", params)
        val response = execute(settings, body, method, version, includeProtocolHeader, mcpName)
        response.optJSONObject("error")?.let { throw McpException(it.optString("message", "MCP request failed")) }
        return response.optJSONObject("result") ?: throw McpException("Invalid MCP response")
    }

    private fun notify(settings: McpSettings, method: String) {
        val body = JSONObject().put("jsonrpc", "2.0").put("method", method)
        execute(settings, body, method, protocolVersion, true, null, expectResponse = false)
    }

    private fun execute(
        settings: McpSettings,
        json: JSONObject,
        method: String,
        version: String,
        includeProtocolHeader: Boolean,
        mcpName: String?,
        expectResponse: Boolean = true
    ): JSONObject {
        val builder = Request.Builder().url(settings.url.trim())
            .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
            .header("Accept", "application/json, text/event-stream")
            .header("Mcp-Method", method)
        if (includeProtocolHeader) builder.header("MCP-Protocol-Version", version)
        if (!mcpName.isNullOrBlank()) builder.header("Mcp-Name", mcpName)
        sessionId?.let { builder.header("Mcp-Session-Id", it) }
        if (settings.token.isNotBlank()) builder.header("Authorization", "Bearer ${settings.token}")

        http.newCall(builder.build()).execute().use { response ->
            response.header("Mcp-Session-Id")?.let { sessionId = it }
            if (!response.isSuccessful) throw McpException("HTTP ${response.code}: ${response.body?.string()?.take(500).orEmpty()}")
            if (!expectResponse || response.code == 202) return JSONObject()
            val raw = response.body?.string().orEmpty()
            if (raw.length > 1_000_000) throw McpException("MCP response is too large")
            val payload = if (response.header("Content-Type").orEmpty().contains("text/event-stream")) {
                raw.lineSequence().filter { it.startsWith("data:") }.map { it.removePrefix("data:").trim() }
                    .lastOrNull { it.contains("\"jsonrpc\"") } ?: throw McpException("Empty MCP event stream")
            } else raw
            return JSONObject(payload)
        }
    }
}

class McpException(message: String) : Exception(message)
