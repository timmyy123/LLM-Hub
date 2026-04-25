package com.llmhub.llmhub.mimobot.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import com.llmhub.llmhub.mimobot.MimoBotIds

/**
 * Phone speaker as an [AudioSink]. Accepts 16 kHz mono Int16 PCM frames and
 * plays them in real time.
 */
class SpeakerSink : AudioSink {

    private val sampleRate = MimoBotIds.SAMPLE_RATE_HZ
    private var track: AudioTrack? = null

    override fun start() {
        if (track != null) return
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(MimoBotIds.FRAME_SAMPLES * 2 * 8)

        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ASSISTANT)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        val format = AudioFormat.Builder()
            .setSampleRate(sampleRate)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .build()

        val t = AudioTrack(
            attrs,
            format,
            minBuf,
            AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        t.play()
        track = t
    }

    override suspend fun write(frame: ShortArray) {
        val t = track ?: run { start(); track!! }
        var offset = 0
        while (offset < frame.size) {
            val n = t.write(frame, offset, frame.size - offset, AudioTrack.WRITE_BLOCKING)
            if (n <= 0) break
            offset += n
        }
    }

    override fun stop() {
        track?.let {
            try { it.pause() } catch (_: Throwable) {}
            try { it.flush() } catch (_: Throwable) {}
            try { it.stop() } catch (_: Throwable) {}
            try { it.release() } catch (_: Throwable) {}
        }
        track = null
    }
}
