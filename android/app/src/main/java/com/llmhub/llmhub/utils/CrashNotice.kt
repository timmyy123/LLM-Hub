package com.llmhub.llmhub.utils

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build

/** Show one recovery hint after a real app crash, including native crashes on Android 11+. */
object CrashNotice {
    private const val PREFS = "crash_notice"
    private const val LAST_LAUNCH = "last_launch"
    private const val LAST_SHOWN_EXIT = "last_shown_exit"
    private const val LEGACY_JAVA_CRASH = "legacy_java_crash"

    fun installJavaCrashHandler(context: Context) {
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean(LEGACY_JAVA_CRASH, true).commit()
            previous?.uncaughtException(thread, error)
        }
    }

    fun consumeOnLaunch(context: Context): Boolean {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val previousLaunch = prefs.getLong(LAST_LAUNCH, 0L)
        val lastShownExit = prefs.getLong(LAST_SHOWN_EXIT, 0L)
        val crashExit = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && previousLaunch > 0L) {
            try {
                val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                manager.getHistoricalProcessExitReasons(context.packageName, 0, 10)
                    .filter { it.processName == context.packageName &&
                        it.timestamp >= previousLaunch && it.timestamp > lastShownExit &&
                        (it.reason == ApplicationExitInfo.REASON_CRASH ||
                            it.reason == ApplicationExitInfo.REASON_CRASH_NATIVE) }
                    .maxOfOrNull { it.timestamp }
            } catch (_: Exception) {
                null
            }
        } else null
        val show = crashExit != null || prefs.getBoolean(LEGACY_JAVA_CRASH, false)
        prefs.edit()
            .putLong(LAST_LAUNCH, System.currentTimeMillis())
            .putLong(LAST_SHOWN_EXIT, crashExit ?: lastShownExit)
            .putBoolean(LEGACY_JAVA_CRASH, false)
            .commit()
        return show
    }
}
