package com.llmhub.llmhub.ui.components

import com.llmhub.llmhub.data.PocketTtsModel
import java.io.Closeable
import java.io.File

/** JNI owns CPU ONNX sessions. Call blocking methods on Dispatchers.IO. */
class PocketTtsEngine(models: File, voices: File) : Closeable {
    fun interface CancellationProbe { fun isCancelled(): Boolean }
    // JNI calls this method by name. Use a concrete class covered by the keep rule,
    // rather than a SAM lambda that R8 can merge with unrelated synthetic classes.
    private class CallbackProbe(private val cancelled: () -> Boolean) : CancellationProbe {
        override fun isCancelled(): Boolean = cancelled()
    }
    private var handle = nativeCreate(models.absolutePath, voices.absolutePath).also { check(it != 0L) }
    @Synchronized fun encode(reference: File) = synchronized(PocketTtsModel.operationLock) {
        check(handle != 0L); nativeEncode(handle, reference.absolutePath)
    }
    @Synchronized fun synthesize(text: String, reference: File, cancelled: () -> Boolean): FloatArray {
        check(handle != 0L)
        return synchronized(PocketTtsModel.operationLock) {
            nativeSynthesize(handle, text, reference.absolutePath, CallbackProbe(cancelled))
        }
    }
    @Synchronized override fun close() { if (handle != 0L) { nativeClose(handle); handle = 0 } }
    private external fun nativeCreate(models: String, voices: String): Long
    private external fun nativeEncode(handle: Long, reference: String)
    private external fun nativeSynthesize(handle: Long, text: String, reference: String, probe: CancellationProbe): FloatArray
    private external fun nativeClose(handle: Long)
    companion object { init { System.loadLibrary("llmhub_pocket_tts") } }
}
