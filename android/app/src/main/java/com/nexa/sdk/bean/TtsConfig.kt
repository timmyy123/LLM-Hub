package com.nexa.sdk.bean

data class TtsConfig(
    val voice: String? = null,
    val speed: Float? = 1.0f,
    val seed: Int? = -1,
    val sampleRate: Int? = 22050
)
