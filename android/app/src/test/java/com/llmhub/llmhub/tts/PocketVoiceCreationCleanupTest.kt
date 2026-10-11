package com.llmhub.llmhub.tts

import com.llmhub.llmhub.ui.components.finishPocketVoiceCreation
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Test
import java.io.IOException

class PocketVoiceCreationCleanupTest {
    @Test fun cancellingCaptureClearsStateAfterIoCleanupAndAllowsRestart() = runBlocking {
        var recording = true
        var busy = true
        var captureJob: Job? = null
        val cleanupStarted = CompletableDeferred<Unit>()
        val allowCleanup = CompletableDeferred<Unit>()
        var deleted = false
        var resets = 0
        captureJob = launch(start = CoroutineStart.UNDISPATCHED) {
            try {
                awaitCancellation()
            } finally {
                finishPocketVoiceCreation(cleanup = {
                    withContext(Dispatchers.IO) {
                        cleanupStarted.complete(Unit)
                        allowCleanup.await()
                        deleted = true
                    }
                }, reset = {
                    assertTrue(deleted)
                    recording = false
                    busy = false
                    captureJob = null
                    resets++
                })
            }
        }
        val cancelledJob = checkNotNull(captureJob)
        cancelledJob.cancel()
        cleanupStarted.await()
        // Keep new capture disabled until the old microphone/file cleanup finishes.
        assertTrue(busy)
        allowCleanup.complete(Unit)
        cancelledJob.join()
        assertFalse(recording)
        assertFalse(busy)
        assertNull(captureJob)
        assertEquals(1, resets)
        captureJob = launch { recording = true }
        captureJob!!.join()
        assertTrue(recording)
    }

    @Test fun cleanupFailureStillResetsState() = runBlocking {
        var reset = false
        try {
            finishPocketVoiceCreation(cleanup = {
                withContext(Dispatchers.IO) { throw IOException("cleanup failed") }
            }, reset = { reset = true })
            fail("Cleanup exception was lost")
        } catch (expected: IOException) {
            assertEquals("cleanup failed", expected.message)
        }
        assertTrue(reset)
    }
}
