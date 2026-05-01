package com.llmhub.llmhub.mimobot.audio

/**
 * Simple energy-based voice activity detection. The device already gates the
 * mic stream with its own VAD, so this is just the phone-side end-of-utterance
 * detector (used when PTT is held to trim trailing silence before handing the
 * buffer to Whisper).
 *
 * TODO(vad): swap to WebRTC VAD (via `com.github.maxhauser:jvad` or a small
 * JNI wrapper) once the end-to-end path works. Energy VAD is good enough for
 * v0 PTT mode — it only decides where to cut, not whether to transmit.
 */
class EnergyVad(
    private val sampleRateHz: Int = 16_000,
    private val thresholdRms: Int = 400,
    private val trailingSilenceMs: Int = 400,
) {
    private var silentMs = 0

    /** Returns true if we have seen [trailingSilenceMs] of silence. */
    fun update(frame: ShortArray): Boolean {
        val rms = rms(frame)
        val frameMs = frame.size * 1000 / sampleRateHz
        silentMs = if (rms < thresholdRms) silentMs + frameMs else 0
        return silentMs >= trailingSilenceMs
    }

    fun reset() { silentMs = 0 }

    private fun rms(frame: ShortArray): Int {
        if (frame.isEmpty()) return 0
        var sumSq = 0L
        for (s in frame) sumSq += s.toLong() * s.toLong()
        return kotlin.math.sqrt(sumSq.toDouble() / frame.size).toInt()
    }
}
