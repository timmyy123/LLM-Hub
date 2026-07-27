package com.llmhub.llmhub.inference

import android.content.Context
import android.content.Intent
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.util.Log
import com.google.ai.edge.litertlm.Tool
import com.google.ai.edge.litertlm.ToolParam
import com.google.ai.edge.litertlm.ToolSet
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import java.net.URL
import java.net.URLEncoder
import java.security.MessageDigest

/**
 * Agent skills toolset for Gemma-4 chat in LLM-Hub.
 *
 * Implements the same set of built-in skills as Google AI Edge Gallery:
 *  - query-wikipedia  : Wikipedia REST API lookup (replaces JS WebView skill)
 *  - calculate-hash   : SHA/MD5 hash computation (replaces JS WebView skill)
 *  - send-email       : Android email intent (matches Gallery "send-email" native skill)
 *  - send-sms         : Android SMS intent (matches Gallery "send-sms" native skill)
 *  - open-map         : Android geo: intent (replaces Gallery "interactive-map" JS/WebView skill)
 *
 * JS/WebView-dependent skills (mood-tracker, qr-code, virtual-piano, text-spinner, etc.)
 * are not implemented here because LLM-Hub's chat UI has no inline WebView renderer.
 */
class ChatAgentSkillsTools(private val context: Context) : ToolSet {

    companion object {
        private const val TAG = "ChatAgentTools"

        /**
         * System instruction injected into ConversationConfig when Gemma-4 is loaded.
         * Mirrors the Gallery's defaultSystemPrompt but adapted for LLM-Hub's tool names.
         */
        val AGENT_SYSTEM_PROMPT: String = """
You are a helpful AI assistant. You have access to the following tools:

- query-wikipedia: Look up factual summaries from Wikipedia. Use for questions about people, places, events, science, history, current affairs, or any topic that benefits from a reliable source.
- calculate-hash: Compute a cryptographic hash (MD5, SHA-1, SHA-256, SHA-512) of any text.
- send-email: Compose and open an email using the device's email app.
- send-sms: Compose and open an SMS message using the device's messaging app.
- open-map: Open a location in the device's map app.
- toggle-flashlight: Turn the device flashlight (camera torch) ON or OFF.

RULES:
1. For factual questions, always use query-wikipedia first to ground your answer.
2. Use tools silently — do NOT narrate intermediate steps or tool calls.
3. After a tool returns a result, use it to compose a concise, direct final answer.
4. If no tool is relevant, answer directly from your knowledge.
5. Never output raw tool call syntax to the user.
        """.trimIndent()
    }

    // ─── Wikipedia ────────────────────────────────────────────────────────────

    @Tool(description = "Query a summary from Wikipedia for a given topic. Use this for factual questions about people, places, events, science, history, or current affairs.")
    fun queryWikipedia(
        @ToolParam(description = "Primary topic to look up (e.g., 'Albert Einstein', '2026 Oscars', 'Great Wall of China'). Extract only the key entity — remove question words and action verbs.") topic: String,
        @ToolParam(description = "2-letter language code that matches the user's language (e.g., 'en', 'es', 'zh', 'fr', 'de', 'ja', 'ko', 'it', 'pt', 'ru', 'ar', 'hi').") lang: String
    ): Map<String, String> {
        return runBlocking(Dispatchers.IO) {
            try {
                val langCode = lang.trim().ifEmpty { "en" }
                val encoded = URLEncoder.encode(topic.trim(), "UTF-8")
                val urlStr = "https://$langCode.wikipedia.org/api/rest_v1/page/summary/$encoded"
                val conn = URL(urlStr).openConnection()
                conn.connectTimeout = 6000
                conn.readTimeout = 6000
                val raw = conn.getInputStream().bufferedReader().use { it.readText() }

                // Parse "extract" field from the JSON response without pulling in a full JSON lib
                val ext = Regex(""""extract"\s*:\s*"((?:[^"\\]|\\.)*)"""")
                    .find(raw)?.groupValues?.get(1)
                    ?.replace("\\n", "\n")?.replace("\\\"", "\"")?.trim()

                if (!ext.isNullOrBlank()) {
                    mapOf("result" to ext, "status" to "succeeded")
                } else {
                    mapOf("error" to "No Wikipedia article found for '$topic'.", "status" to "failed")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Wikipedia lookup failed for '$topic': ${e.message}")
                mapOf("error" to "Wikipedia lookup failed: ${e.message}", "status" to "failed")
            }
        }
    }

    // ─── Hash ─────────────────────────────────────────────────────────────────

    @Tool(description = "Calculate the cryptographic hash of a piece of text.")
    fun calculateHash(
        @ToolParam(description = "The text to hash.") text: String,
        @ToolParam(description = "Hash algorithm to use: MD5, SHA-1, SHA-256 (default), or SHA-512.") algorithm: String
    ): Map<String, String> {
        return try {
            val algo = when (algorithm.trim().uppercase()) {
                "SHA1", "SHA-1" -> "SHA-1"
                "SHA512", "SHA-512" -> "SHA-512"
                "MD5" -> "MD5"
                else -> "SHA-256"
            }
            val hash = MessageDigest.getInstance(algo)
                .digest(text.toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it) }
            mapOf("result" to hash, "algorithm" to algo, "status" to "succeeded")
        } catch (e: Exception) {
            mapOf("error" to "Hash calculation failed: ${e.message}", "status" to "failed")
        }
    }

    // ─── Email ────────────────────────────────────────────────────────────────

    @Tool(description = "Send an email using the device's email app. Opens a pre-filled compose window.")
    fun sendEmail(
        @ToolParam(description = "Recipient email address.") email: String,
        @ToolParam(description = "Subject line of the email.") subject: String,
        @ToolParam(description = "Body text of the email.") body: String
    ): Map<String, String> {
        return try {
            val intent = Intent(Intent.ACTION_SEND).apply {
                data = Uri.parse("mailto:")
                type = "text/plain"
                putExtra(Intent.EXTRA_EMAIL, arrayOf(email.trim()))
                putExtra(Intent.EXTRA_SUBJECT, subject)
                putExtra(Intent.EXTRA_TEXT, body)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            mapOf("result" to "Email app opened. To: $email, Subject: $subject", "status" to "succeeded")
        } catch (e: Exception) {
            Log.w(TAG, "sendEmail failed: ${e.message}")
            mapOf("error" to "Could not open email app: ${e.message}", "status" to "failed")
        }
    }

    // ─── SMS ──────────────────────────────────────────────────────────────────

    @Tool(description = "Send an SMS text message using the device's messaging app. Opens a pre-filled compose window.")
    fun sendSms(
        @ToolParam(description = "Recipient phone number (digits only, e.g., '14155552671').") phoneNumber: String,
        @ToolParam(description = "Body text of the SMS message.") body: String
    ): Map<String, String> {
        return try {
            val uri = Uri.parse("smsto:${phoneNumber.trim()}")
            val intent = Intent(Intent.ACTION_SENDTO, uri).apply {
                putExtra("sms_body", body)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            mapOf("result" to "SMS app opened. To: $phoneNumber", "status" to "succeeded")
        } catch (e: Exception) {
            Log.w(TAG, "sendSms failed: ${e.message}")
            mapOf("error" to "Could not open SMS app: ${e.message}", "status" to "failed")
        }
    }

    // ─── Map ──────────────────────────────────────────────────────────────────

    @Tool(description = "Show a location on the device's map app (Google Maps or any installed map).")
    fun openMap(
        @ToolParam(description = "Location to display (e.g., 'Eiffel Tower, Paris', 'Googleplex, Mountain View', or coordinates '37.4219983,-122.084').") location: String
    ): Map<String, String> {
        return try {
            val encoded = URLEncoder.encode(location.trim(), "UTF-8")
            val uri = Uri.parse("geo:0,0?q=$encoded")
            val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            mapOf("result" to "Map opened for '$location'.", "status" to "succeeded")
        } catch (e: Exception) {
            Log.w(TAG, "openMap failed: ${e.message}")
            mapOf("error" to "Could not open map app: ${e.message}", "status" to "failed")
        }
    }

    // ─── Flashlight ───────────────────────────────────────────────────────────

    @Tool(description = "Turn the device flashlight (camera torch) ON or OFF.")
    fun toggleFlashlight(
        @ToolParam(description = "Set to 'true' to turn flashlight ON, or 'false' to turn flashlight OFF.") enabled: String
    ): Map<String, String> {
        val isEnabled = enabled.lowercase() == "true" || enabled.lowercase() == "on" || enabled == "1"
        return try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = cameraManager.cameraIdList.firstOrNull()
            if (cameraId != null) {
                cameraManager.setTorchMode(cameraId, isEnabled)
                mapOf("result" to "Flashlight turned ${if (isEnabled) "on" else "off"}.", "status" to "succeeded")
            } else {
                mapOf("error" to "No camera available for flashlight.", "status" to "failed")
            }
        } catch (e: Exception) {
            Log.w(TAG, "toggleFlashlight failed: ${e.message}")
            mapOf("error" to "Flashlight error: ${e.message}", "status" to "failed")
        }
    }
}
