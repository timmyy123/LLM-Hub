package com.llmhub.llmhub.mimobot.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import com.llmhub.llmhub.mimobot.MimoBotIds
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.Dispatchers
import kotlin.concurrent.thread

/**
 * Phone microphone as an [AudioSource]. Captures 16 kHz mono PCM and emits
 * [MimoBotIds.FRAME_SAMPLES]-sample frames.
 *
 * Caller is responsible for holding the RECORD_AUDIO runtime permission before
 * starting the flow.
 */
class MicSource : AudioSource {

    private val frameSamples = MimoBotIds.FRAME_SAMPLES
    private val sampleRate = MimoBotIds.SAMPLE_RATE_HZ

    @Volatile private var record: AudioRecord? = null
    @Volatile private var running = false

    @SuppressLint("MissingPermission")
    override fun frames(): Flow<ShortArray> = callbackFlow {
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(frameSamples * 2 * 4)

        val ar = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf,
        )
        if (ar.state != AudioRecord.STATE_INITIALIZED) {
            close(IllegalStateException("AudioRecord failed to initialize"))
            return@callbackFlow
        }

        record = ar
        running = true
        ar.startRecording()

        val readerThread = thread(name = "mimo-mic", isDaemon = true) {
            val buf = ShortArray(frameSamples)
            try {
                while (running && ar.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    var read = 0
                    while (read < frameSamples && running) {
                        val n = ar.read(buf, read, frameSamples - read)
                        if (n <= 0) break
                        read += n
                    }
                    if (read == frameSamples) {
                        // Copy so downstream can hold it safely while we reuse buf.
                        val frame = buf.copyOf()
                        trySend(frame)
                    }
                }
            } catch (t: Throwable) {
                if (t !is CancellationException) Log.w("MicSource", "reader: ${t.message}")
                close(t)
            }
        }

        awaitClose {
            running = false
            try { ar.stop() } catch (_: Throwable) {}
            try { ar.release() } catch (_: Throwable) {}
            record = null
            // readerThread will exit on `running = false`
            try { readerThread.join(200) } catch (_: Throwable) {}
        }
    }.flowOn(Dispatchers.IO)

    override fun stop() {
        running = false
        record?.let {
            try { it.stop() } catch (_: Throwable) {}
            try { it.release() } catch (_: Throwable) {}
        }
        record = null
    }
}
