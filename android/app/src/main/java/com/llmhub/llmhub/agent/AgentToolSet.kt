package com.llmhub.llmhub.agent

import android.content.Context
import android.content.Intent
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.provider.AlarmClock
import android.provider.CalendarContract
import android.util.Log
import com.google.ai.edge.litertlm.Tool
import com.google.ai.edge.litertlm.ToolParam
import com.google.ai.edge.litertlm.ToolSet
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import java.net.URL
import java.net.URLEncoder
import java.security.MessageDigest

/**
 * Complete ToolSet for the standalone Agent feature in LLM-Hub.
 */
class AgentToolSet(
    private val context: Context,
    var isTermuxEnabled: Boolean = false
) : ToolSet {

    companion object {
        private const val TAG = "AgentToolSet"

        val AGENT_SYSTEM_PROMPT = """
You are a smart AI Agent with direct tool execution capabilities. Available tools:

- show_map: Find and display a place or address on the in-app interactive map. Location can be a place name or address.
- query_wikipedia: Look up factual summaries from Wikipedia.
- web_search: Search the live web for recent information or general queries.
- get_current_weather: Get current weather for any city or location.
- create_calendar_event: Create an event in the device calendar with title, location, start and end time.
- set_alarm: Set an alarm on the device for a specific hour and minute.
- send_email: Compose an email to a recipient.
- send_sms: Compose an SMS message to a phone number.
- calculate_hash: Calculate cryptographic hash (MD5, SHA-256, etc.) of text.
- toggle_flashlight: Turn the device flashlight (camera torch) ON or OFF.
- run_termux_command: Run a shell command in Termux (if enabled).

RULES:
1. When asked about places, weather, news, calculations, calendar, alarms, or emails, use the appropriate tool.
2. Formulate concise tool parameters.
3. Incorporate tool execution results cleanly into your final response.
        """.trimIndent()
    }

    // ─── Show Map (Nominatim Geocoding + In-App Map Data) ─────────────────────

    @Tool(description = "Find and show a place or address on an interactive map. Returns coordinates and place details.")
    fun showMap(
        @ToolParam(description = "Location, landmark or address to find (e.g. 'Eiffel Tower, Paris', 'Times Square, New York', 'Tokyo Tower').") location: String
    ): Map<String, Any> {
        return runBlocking(Dispatchers.IO) {
            try {
                val encoded = URLEncoder.encode(location.trim(), "UTF-8")
                val urlStr = "https://nominatim.openstreetmap.org/search?q=$encoded&format=json&limit=1"
                val conn = URL(urlStr).openConnection()
                conn.setRequestProperty("User-Agent", "LLMHub-Agent/1.0")
                conn.connectTimeout = 8000
                conn.readTimeout = 8000
                val raw = conn.getInputStream().bufferedReader().use { it.readText() }
                val jsonArr = JSONArray(raw)

                if (jsonArr.length() > 0) {
                    val obj = jsonArr.getJSONObject(0)
                    val lat = obj.getDouble("lat")
                    val lon = obj.getDouble("lon")
                    val displayName = obj.optString("display_name", location)
                    mapOf(
                        "type" to "map",
                        "lat" to lat,
                        "lon" to lon,
                        "label" to displayName,
                        "status" to "succeeded"
                    )
                } else {
                    openGeoIntent(location)
                    mapOf("type" to "map_intent", "location" to location, "status" to "succeeded")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Map geocoding failed for '$location': ${e.message}")
                openGeoIntent(location)
                mapOf("type" to "map_intent", "location" to location, "status" to "succeeded")
            }
        }
    }

    private fun openGeoIntent(location: String) {
        try {
            val encoded = URLEncoder.encode(location.trim(), "UTF-8")
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse("geo:0,0?q=$encoded")).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
        } catch (_: Exception) {}
    }

    // ─── Query Wikipedia ──────────────────────────────────────────────────────

    @Tool(description = "Query a summary from Wikipedia for a given topic. Useful for facts, history, science, and people.")
    fun queryWikipedia(
        @ToolParam(description = "Topic to look up.") topic: String,
        @ToolParam(description = "2-letter language code (default 'en').") lang: String
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

                val ext = Regex(""""extract"\s*:\s*"((?:[^"\\]|\\.)*)"""")
                    .find(raw)?.groupValues?.get(1)
                    ?.replace("\\n", "\n")?.replace("\\\"", "\"")?.trim()

                if (!ext.isNullOrBlank()) {
                    mapOf("result" to ext, "status" to "succeeded")
                } else {
                    mapOf("error" to "No article found for '$topic'.", "status" to "failed")
                }
            } catch (e: Exception) {
                mapOf("error" to "Wikipedia lookup failed: ${e.message}", "status" to "failed")
            }
        }
    }

    // ─── Web Search ───────────────────────────────────────────────────────────

    @Tool(description = "Perform a live web search using DuckDuckGo to get up-to-date facts or web snippets.")
    fun webSearch(
        @ToolParam(description = "Search query.") query: String
    ): Map<String, String> {
        return runBlocking(Dispatchers.IO) {
            try {
                val encoded = URLEncoder.encode(query.trim(), "UTF-8")
                val urlStr = "https://html.duckduckgo.com/html/?q=$encoded"
                val conn = URL(urlStr).openConnection()
                conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                conn.connectTimeout = 8000
                conn.readTimeout = 8000
                val html = conn.getInputStream().bufferedReader().use { it.readText() }

                val snippets = Regex("""<a class="result__snippet[^">]*>(.*?)</a>""")
                    .findAll(html)
                    .take(3)
                    .map { it.groupValues[1].replace(Regex("<.*?>"), "").trim() }
                    .filter { it.isNotEmpty() }
                    .joinToString("\n---\n")

                if (snippets.isNotBlank()) {
                    mapOf("result" to snippets, "status" to "succeeded")
                } else {
                    mapOf("result" to "No search results found for query.", "status" to "succeeded")
                }
            } catch (e: Exception) {
                mapOf("error" to "Web search failed: ${e.message}", "status" to "failed")
            }
        }
    }

    // ─── Weather ──────────────────────────────────────────────────────────────

    @Tool(description = "Get current weather conditions and forecast for a location.")
    fun getCurrentWeather(
        @ToolParam(description = "City or location name (e.g., 'London', 'Tokyo', 'Berlin').") location: String
    ): Map<String, String> {
        return runBlocking(Dispatchers.IO) {
            try {
                val encoded = URLEncoder.encode(location.trim(), "UTF-8")
                val urlStr = "https://wttr.in/$encoded?format=3"
                val conn = URL(urlStr).openConnection()
                conn.connectTimeout = 6000
                conn.readTimeout = 6000
                val text = conn.getInputStream().bufferedReader().use { it.readText().trim() }
                if (text.isNotBlank()) {
                    mapOf("result" to text, "status" to "succeeded")
                } else {
                    mapOf("error" to "Weather unavailable for '$location'.", "status" to "failed")
                }
            } catch (e: Exception) {
                mapOf("error" to "Weather request failed: ${e.message}", "status" to "failed")
            }
        }
    }

    // ─── Calendar Event ───────────────────────────────────────────────────────

    @Tool(description = "Create a new event in the device calendar.")
    fun createCalendarEvent(
        @ToolParam(description = "Event title or summary.") title: String,
        @ToolParam(description = "Location of the event (optional).") location: String,
        @ToolParam(description = "Description or notes for the event (optional).") description: String
    ): Map<String, String> {
        return try {
            val intent = Intent(Intent.ACTION_INSERT).apply {
                data = CalendarContract.Events.CONTENT_URI
                putExtra(CalendarContract.Events.TITLE, title)
                if (location.isNotBlank()) putExtra(CalendarContract.Events.EVENT_LOCATION, location)
                if (description.isNotBlank()) putExtra(CalendarContract.Events.DESCRIPTION, description)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            mapOf("result" to "Opened Calendar to create event: '$title'", "status" to "succeeded")
        } catch (e: Exception) {
            mapOf("error" to "Could not open Calendar: ${e.message}", "status" to "failed")
        }
    }

    // ─── Set Alarm ────────────────────────────────────────────────────────────

    @Tool(description = "Set an alarm on the device for a specific hour and minute.")
    fun setAlarm(
        @ToolParam(description = "Hour of the alarm (0 to 23).") hour: Int,
        @ToolParam(description = "Minute of the alarm (0 to 59).") minute: Int,
        @ToolParam(description = "Label or name for the alarm.") label: String
    ): Map<String, String> {
        return try {
            val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                putExtra(AlarmClock.EXTRA_HOUR, hour)
                putExtra(AlarmClock.EXTRA_MINUTES, minute)
                putExtra(AlarmClock.EXTRA_MESSAGE, label)
                putExtra(AlarmClock.EXTRA_SKIP_UI, false)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            mapOf("result" to "Alarm set for %02d:%02d ('$label')".format(hour, minute), "status" to "succeeded")
        } catch (e: Exception) {
            mapOf("error" to "Could not set alarm: ${e.message}", "status" to "failed")
        }
    }

    // ─── Send Email ───────────────────────────────────────────────────────────

    @Tool(description = "Open email app with pre-filled recipient, subject, and body.")
    fun sendEmail(
        @ToolParam(description = "Recipient email address.") email: String,
        @ToolParam(description = "Subject line.") subject: String,
        @ToolParam(description = "Body message.") body: String
    ): Map<String, String> {
        return try {
            val intent = Intent(Intent.ACTION_SENDTO).apply {
                data = Uri.parse("mailto:")
                putExtra(Intent.EXTRA_EMAIL, arrayOf(email.trim()))
                putExtra(Intent.EXTRA_SUBJECT, subject)
                putExtra(Intent.EXTRA_TEXT, body)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            mapOf("result" to "Opened Email app for recipient '$email'", "status" to "succeeded")
        } catch (e: Exception) {
            mapOf("error" to "Could not open Email app: ${e.message}", "status" to "failed")
        }
    }

    // ─── Send SMS ─────────────────────────────────────────────────────────────

    @Tool(description = "Open SMS app with pre-filled phone number and message.")
    fun sendSms(
        @ToolParam(description = "Phone number.") phoneNumber: String,
        @ToolParam(description = "SMS message content.") body: String
    ): Map<String, String> {
        return try {
            val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:${phoneNumber.trim()}")).apply {
                putExtra("sms_body", body)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            mapOf("result" to "Opened SMS app for '$phoneNumber'", "status" to "succeeded")
        } catch (e: Exception) {
            mapOf("error" to "Could not open SMS app: ${e.message}", "status" to "failed")
        }
    }

    // ─── Calculate Hash ───────────────────────────────────────────────────────

    @Tool(description = "Calculate cryptographic hash of input text.")
    fun calculateHash(
        @ToolParam(description = "Text to hash.") text: String,
        @ToolParam(description = "Algorithm: MD5, SHA-1, SHA-256, SHA-512.") algorithm: String
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

    // ─── Termux Command ───────────────────────────────────────────────────────

    @Tool(description = "Execute a terminal shell command in Termux (if enabled in settings).")
    fun runTermuxCommand(
        @ToolParam(description = "Command string to execute.") command: String
    ): Map<String, String> {
        if (!isTermuxEnabled) {
            return mapOf("error" to "Termux commands are currently disabled in Agent settings.", "status" to "failed")
        }
        return try {
            val intent = Intent().apply {
                setClassName("com.termux", "com.termux.app.RunCommandService")
                action = "com.termux.RUN_COMMAND"
                putExtra("com.termux.RUN_COMMAND_PATH", "/data/data/com.termux/files/usr/bin/bash")
                putExtra("com.termux.RUN_COMMAND_ARGUMENTS", arrayOf("-c", command))
                putExtra("com.termux.RUN_COMMAND_BACKGROUND", true)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startService(intent)
            mapOf("result" to "Sent command to Termux: '$command'", "status" to "succeeded")
        } catch (e: Exception) {
            mapOf("error" to "Termux execution failed: ${e.message}", "status" to "failed")
        }
    }

    // ─── Flashlight ───────────────────────────────────────────────────────────

    @Tool(description = "Turn the device flashlight (camera torch) ON or OFF.")
    fun toggleFlashlight(
        @ToolParam(description = "Set to 'true' to turn flashlight ON, or 'false' to turn flashlight OFF.") enabled: String
    ): Map<String, Any> {
        val isEnabled = enabled.lowercase() == "true" || enabled.lowercase() == "on" || enabled == "1"
        return try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = cameraManager.cameraIdList.firstOrNull()
            if (cameraId != null) {
                cameraManager.setTorchMode(cameraId, isEnabled)
                mapOf("status" to "succeeded", "result" to "Flashlight turned ${if (isEnabled) "on" else "off"}.")
            } else {
                mapOf("status" to "failed", "error" to "No camera available for flashlight.")
            }
        } catch (e: Exception) {
            mapOf("status" to "failed", "error" to "Flashlight error: ${e.message}")
        }
    }
}
