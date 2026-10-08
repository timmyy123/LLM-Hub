package com.llmhub.llmhub.agent

import android.content.Context
import kotlinx.coroutines.*
import kotlinx.coroutines.selects.select

internal object TermuxStreamingCommand {
    suspend fun run(context: Context, command: String, timeoutMs: Long, onOutput: (String) -> Unit): String =
        withContext(Dispatchers.IO) {
            val stream = TermuxOutputStream()
            try {
                supervisorScope {
                    val result = async {
                        try { Result.success(TermuxCommandBridge.run(context, stream.commandScript(command), timeoutMs)) }
                        catch (e: CancellationException) { throw e }
                        catch (e: Exception) { Result.failure(e) }
                    }
                    val output = async { stream.collect(onOutput) }
                    var completed = false
                    try {
                        val value = withTimeout(timeoutMs) {
                            select<String> {
                                result.onAwait { final ->
                                    // Drain the last bytes even when RUN_COMMAND reports a nonzero exit.
                                    withTimeoutOrNull(1500) { output.await() }
                                    final.getOrThrow()
                                }
                                output.onAwait { result.await().getOrThrow() }
                            }
                        }
                        completed = true
                        value
                    } finally {
                        withContext(NonCancellable) {
                            if (!completed) stream.processGroup?.let { group ->
                                runCatching { TermuxCommandBridge.run(context,
                                    "kill -TERM -- -$group 2>/dev/null || true", 5000) }
                            }
                            stream.close()
                            output.cancelAndJoin()
                            result.cancelAndJoin()
                        }
                    }
                }
            } finally { stream.close() }
        }
}
