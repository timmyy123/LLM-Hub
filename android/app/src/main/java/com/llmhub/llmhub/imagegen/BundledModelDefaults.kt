package com.llmhub.llmhub.imagegen

import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Sampler settings a QNN model zip ships in its own config.json (local-dream format), e.g. DMD2 SDXL
 * `{"default_scheduler":"lcm","default_steps":8,"default_cfg":1}` or Anima Turbo `euler / 10 / 1`.
 * These distilled models burn when run with the app's usual DPM++ and guidance 7, so these values win.
 */
data class BundledModelDefaults(val scheduler: String, val steps: Int?, val cfg: Float?) {
    companion object {
        private const val TAG = "BundledModelDefaults"
        private const val MAX_DEPTH = 6

        /** Scheduler names libstable_diffusion_core.so accepts. */
        private val SCHEDULERS = setOf("dpm", "dpm_karras", "dpm_sde_karras", "dpmsolver++", "euler", "euler_a", "lcm")

        /** Reads the config.json inside a downloaded model folder, or null when the model ships none. */
        fun read(modelPath: String?): BundledModelDefaults? {
            val root = modelPath?.let(::File)?.takeIf { it.isDirectory } ?: return null
            val file = find(root, 0) ?: return null
            return try {
                parse(file.readText())
            } catch (e: Exception) {
                Log.w(TAG, "Unreadable ${file.path}: ${e.message}")
                null
            }
        }

        fun parse(text: String): BundledModelDefaults? {
            val json = JSONObject(text)
            val scheduler = json.optString("default_scheduler").lowercase().takeIf { it in SCHEDULERS } ?: return null
            val steps = json.optInt("default_steps", 0).takeIf { it > 0 }
            val cfg = json.optDouble("default_cfg", Double.NaN).takeIf { !it.isNaN() && it > 0 }?.toFloat()
            return BundledModelDefaults(scheduler, steps, cfg)
        }

        private fun find(dir: File, depth: Int): File? {
            if (depth > MAX_DEPTH) return null
            val children = dir.listFiles() ?: return null
            children.firstOrNull { it.isFile && it.name == "config.json" && it.length() < 64_000 && it.readText().contains("default_scheduler") }
                ?.let { return it }
            return children.filter { it.isDirectory }.firstNotNullOfOrNull { find(it, depth + 1) }
        }
    }
}
