package com.llmhub.llmhub.ui.components

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/** Foreground, bounded PCM capture. The caller must obtain microphone permission. */
internal object PocketVoiceRecorder {
    const val SAMPLE_RATE = 24000
    const val MAX_SECONDS = 30
    @SuppressLint("MissingPermission")
    suspend fun record(file: File, stop: AtomicBoolean, progress: (Int) -> Unit) = withContext(Dispatchers.IO) {
        val minimum = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        check(minimum > 0)
        val recorder = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
            .setAudioFormat(AudioFormat.Builder().setSampleRate(SAMPLE_RATE)
                .setChannelMask(AudioFormat.CHANNEL_IN_MONO).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
            .setBufferSizeInBytes(maxOf(minimum * 2, 9600)).build()
        try {
            check(recorder.state == AudioRecord.STATE_INITIALIZED)
            recorder.startRecording()
            check(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING)
            val pcm = ByteArrayOutputStream()
            val buffer = ByteArray(4800)
            val limit = SAMPLE_RATE * 2 * MAX_SECONDS
            var lastSecond = -1
            val deadline = android.os.SystemClock.elapsedRealtime() + 35000L
            while (!stop.get() && pcm.size() < limit) {
                currentCoroutineContext().ensureActive()
                check(android.os.SystemClock.elapsedRealtime() < deadline)
                val read = recorder.read(buffer, 0, minOf(buffer.size, limit - pcm.size()), AudioRecord.READ_NON_BLOCKING)
                check(read >= 0) { "Microphone read failed: $read" }
                if (read > 0) pcm.write(buffer, 0, read) else delay(10)
                val seconds = pcm.size() / (SAMPLE_RATE * 2)
                if (seconds != lastSecond) {
                    lastSecond = seconds
                    withContext(Dispatchers.Main) { progress(seconds) }
                }
            }
            currentCoroutineContext().ensureActive()
            require(pcm.size() >= SAMPLE_RATE * 2 * 3) { "Recording too short" }
            val audio = pcm.toByteArray()
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
                put("RIFF".toByteArray()); putInt(36 + audio.size); put("WAVEfmt ".toByteArray()); putInt(16)
                putShort(1); putShort(1); putInt(SAMPLE_RATE); putInt(SAMPLE_RATE * 2); putShort(2); putShort(16)
                put("data".toByteArray()); putInt(audio.size)
            }.array()
            file.outputStream().use { it.write(header); it.write(audio) }
        } finally {
            try { if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) recorder.stop() }
            finally { recorder.release() }
        }
    }
}
