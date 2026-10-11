package com.llmhub.llmhub.ui.components

import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext

/** Keep IO cleanup and the UI reset under the same cancellation-safe context. */
internal suspend fun finishPocketVoiceCreation(cleanup: suspend () -> Unit, reset: () -> Unit) {
    withContext(NonCancellable) {
        try {
            cleanup()
        } finally {
            reset()
        }
    }
}
