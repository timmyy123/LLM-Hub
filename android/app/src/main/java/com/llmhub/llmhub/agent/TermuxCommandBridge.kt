package com.llmhub.llmhub.agent

import android.app.Activity
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.IBinder
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.withTimeout
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

internal const val TERMUX_RUN_COMMAND_PERMISSION = "com.termux.permission.RUN_COMMAND"
private const val TERMUX_PACKAGE = "com.termux"
private const val TERMUX_SERVICE = "com.termux.app.RunCommandService"

internal fun isTermuxRunCommandAvailable(context: Context): Boolean {
    val permissionExists = runCatching {
        context.packageManager.getPermissionInfo(TERMUX_RUN_COMMAND_PERMISSION, 0)
    }.isSuccess
    if (!permissionExists) return false
    val intent = Intent("com.termux.RUN_COMMAND").setClassName(TERMUX_PACKAGE, TERMUX_SERVICE)
    return context.packageManager.resolveService(intent, 0) != null
}

internal data class TermuxCommandResult(
    val stdout: String,
    val stderr: String,
    val exitCode: Int,
    val errorCode: Int,
    val errorMessage: String
)

/**
 * Executes a command using Termux's documented RUN_COMMAND PendingIntent API.
 * Results cannot be read from LLM Hub's external-files directory because Termux
 * is a different Android app and scoped storage blocks that directory.
 */
internal object TermuxCommandBridge {
    private const val EXTRA_EXECUTION_ID = "llmhub_termux_execution_id"
    private val nextExecutionId = AtomicInteger(1000)
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<TermuxCommandResult>>()

    suspend fun run(context: Context, command: String, timeoutMs: Long): String {
        if (!isTermuxRunCommandAvailable(context)) {
            throw McpException("termux_run_command_unavailable")
        }
        if (ContextCompat.checkSelfPermission(context, TERMUX_RUN_COMMAND_PERMISSION) != PackageManager.PERMISSION_GRANTED) {
            throw McpException("termux_permission_required")
        }

        val executionId = nextExecutionId.getAndIncrement()
        val result = CompletableDeferred<TermuxCommandResult>()
        pending[executionId] = result

        val callbackIntent = Intent(context, TermuxResultService::class.java)
            .putExtra(EXTRA_EXECUTION_ID, executionId)
        val callbackFlags = PendingIntent.FLAG_ONE_SHOT or
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_MUTABLE else 0
        val callback = PendingIntent.getService(context, executionId, callbackIntent, callbackFlags)

        val runIntent = Intent().apply {
            setClassName(TERMUX_PACKAGE, TERMUX_SERVICE)
            action = "com.termux.RUN_COMMAND"
            putExtra("com.termux.RUN_COMMAND_PATH", "/data/data/com.termux/files/usr/bin/bash")
            putExtra("com.termux.RUN_COMMAND_ARGUMENTS", arrayOf("-c", command))
            putExtra("com.termux.RUN_COMMAND_WORKDIR", "/data/data/com.termux/files/home")
            putExtra("com.termux.RUN_COMMAND_BACKGROUND", true)
            putExtra("com.termux.RUN_COMMAND_PENDING_INTENT", callback)
        }

        try {
            // Modern Android blocks cross-app background service starts even while
            // LLM Hub is visible on some OEM builds. Termux promotes command work
            // through its foreground service, so request a foreground start here.
            ContextCompat.startForegroundService(context, runIntent)
            val commandResult = withTimeout(timeoutMs) { result.await() }
            if (commandResult.errorCode != Activity.RESULT_OK || commandResult.exitCode != 0) {
                val detail = commandResult.errorMessage.ifBlank { commandResult.stderr }
                    .ifBlank { "Termux command failed (${commandResult.exitCode})" }
                throw McpException(detail.takeLast(2_000))
            }
            return commandResult.stdout.take(1_000_000)
        } finally {
            pending.remove(executionId)
            callback.cancel()
        }
    }

    fun complete(executionId: Int, result: TermuxCommandResult) {
        pending.remove(executionId)?.complete(result)
    }

    fun executionId(intent: Intent): Int = intent.getIntExtra(EXTRA_EXECUTION_ID, 0)
}

/** Receives the result PendingIntent sent by Termux 0.109 or newer. */
class TermuxResultService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent != null) {
            val bundle = intent.getBundleExtra("result")
            val executionId = TermuxCommandBridge.executionId(intent)
            if (bundle != null && executionId != 0) {
                TermuxCommandBridge.complete(
                    executionId,
                    TermuxCommandResult(
                        stdout = bundle.getString("stdout", ""),
                        stderr = bundle.getString("stderr", ""),
                        exitCode = bundle.getInt("exitCode", -1),
                        errorCode = bundle.getInt("err", Activity.RESULT_OK),
                        errorMessage = bundle.getString("errmsg", "")
                    )
                )
            }
        }
        stopSelf(startId)
        return START_NOT_STICKY
    }
}
