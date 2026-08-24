package com.llmhub.llmhub.inference

import android.content.Context
import android.os.Build

enum class GgufEngine {
    DEFAULT_GENIEX,
    LLAMA_CPP,
}

/** Shared GGUF routing preference and native-load crash marker. */
object GgufEnginePolicy {
    private const val PREFS_NAME = "gguf_engine_preferences"
    private const val CRASH_PREFS_NAME = "geniex_native_crash_guard"
    private const val ENGINE_KEY = "selected_engine"
    private val LLAMA_CPP_SOCS = setOf("SM8450")

    fun selectedEngine(context: Context): GgufEngine {
        val stored = preferences(context).getString(ENGINE_KEY, null)
        return runCatching { GgufEngine.valueOf(stored.orEmpty()) }
            .getOrDefault(GgufEngine.DEFAULT_GENIEX)
    }

    fun setSelectedEngine(context: Context, engine: GgufEngine) {
        preferences(context).edit().putString(ENGINE_KEY, engine.name).apply()
    }

    fun shouldUseLlamaCpp(context: Context): Boolean {
        return deviceSoc() in LLAMA_CPP_SOCS || selectedEngine(context) == GgufEngine.LLAMA_CPP
    }

    fun deviceSoc(): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.orEmpty().uppercase()
        } else {
            Build.HARDWARE.orEmpty().uppercase()
        }
    }

    /** Must be committed before vendor JNI because SIGABRT skips Kotlin catch/finally. */
    fun markGeniexLoadStarted(context: Context) {
        crashPreferences(context).edit().putBoolean(crashMarkerKey(context), true).commit()
    }

    fun markGeniexLoadFinished(context: Context) {
        crashPreferences(context).edit().putBoolean(crashMarkerKey(context), false).commit()
    }

    fun hasInterruptedGeniexLoad(context: Context): Boolean {
        return crashPreferences(context).getBoolean(crashMarkerKey(context), false)
    }

    fun acknowledgeInterruptedGeniexLoad(context: Context) {
        crashPreferences(context).edit().putBoolean(crashMarkerKey(context), false).apply()
    }

    private fun preferences(context: Context) =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private fun crashPreferences(context: Context) =
        context.applicationContext.getSharedPreferences(CRASH_PREFS_NAME, Context.MODE_PRIVATE)

    private fun crashMarkerKey(context: Context): String {
        val versionCode = runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode
            } else {
                @Suppress("DEPRECATION")
                context.packageManager.getPackageInfo(context.packageName, 0).versionCode.toLong()
            }
        }.getOrDefault(0L)
        return "load_in_progress_${versionCode}_${deviceSoc().ifBlank { "unknown" }}"
    }
}
