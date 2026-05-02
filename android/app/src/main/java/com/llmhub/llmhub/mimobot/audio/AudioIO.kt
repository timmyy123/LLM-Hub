package com.llmhub.llmhub.mimobot.audio

import kotlinx.coroutines.flow.Flow

/**
 * Abstractions that let [com.llmhub.llmhub.mimobot.pipeline.VoicePipeline] run
 * against either the phone's own mic+speaker (local dev mode) or a BLE
 * transport with Opus codec (real device mode).
 *
 * Contract: all frames are 16 kHz mono Int16 PCM, 320 samples (20 ms) each.
 */
interface AudioSource {
    /** Start capturing; flow completes when [stop] is called. */
    fun frames(): Flow<ShortArray>
    fun stop()
}

interface AudioSink {
    /** Must be called before [write]. Idempotent. */
    fun start()
    suspend fun write(frame: ShortArray)
    fun stop()
}
