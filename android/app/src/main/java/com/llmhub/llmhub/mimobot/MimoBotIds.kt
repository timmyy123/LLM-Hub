package com.llmhub.llmhub.mimobot

import java.util.UUID

/**
 * Canonical identifiers shared across the Mimo bot stack.
 * Must stay in sync with docs/mimobot-protocol.md and firmware/main/ble_transport.c.
 */
object MimoBotIds {
    val SERVICE: UUID    = UUID.fromString("6d696d6f-b07d-4e13-9e88-000000000001")
    val AUDIO_UP: UUID   = UUID.fromString("6d696d6f-b07d-4e13-9e88-000000000010")
    val AUDIO_DOWN: UUID = UUID.fromString("6d696d6f-b07d-4e13-9e88-000000000011")
    val CONTROL: UUID    = UUID.fromString("6d696d6f-b07d-4e13-9e88-000000000012")

    const val ADV_NAME_PREFIX = "mimobot-"

    // Audio format constants (see docs/mimobot-protocol.md)
    const val SAMPLE_RATE_HZ = 16_000
    const val FRAME_MS = 20
    const val FRAME_SAMPLES = SAMPLE_RATE_HZ * FRAME_MS / 1000  // 320
    const val CHANNELS = 1
    const val OPUS_BITRATE = 24_000
}
