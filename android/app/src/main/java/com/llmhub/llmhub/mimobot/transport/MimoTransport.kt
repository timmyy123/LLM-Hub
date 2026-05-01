package com.llmhub.llmhub.mimobot.transport

import kotlinx.coroutines.flow.Flow

/**
 * Abstraction over the link to the Mimo bot. v0 is always BLE; later transports
 * (Wi-Fi UDP, BT Classic) implement the same interface so the pipeline above
 * doesn't care.
 *
 * Frames on [audioUp] and [audioDown] are Opus payloads — sequence numbers
 * and BLE framing are handled below this layer.
 */
interface MimoTransport {

    /** Connection state reported to UI / pipeline. */
    enum class State { DISCONNECTED, SCANNING, CONNECTING, CONNECTED, ERROR }

    val state: Flow<State>

    /** Opus frames received from the device (mic). */
    val audioUp: Flow<ByteArray>

    /** JSON control messages received from the device (see protocol spec). */
    val control: Flow<String>

    /** Connect to the first discovered mimobot-* advertisement. */
    suspend fun connect()

    /** Send an Opus frame to the device (speaker). */
    suspend fun sendAudioDown(opusFrame: ByteArray)

    /** Send a JSON control message to the device. */
    suspend fun sendControl(json: String)

    suspend fun disconnect()
}
