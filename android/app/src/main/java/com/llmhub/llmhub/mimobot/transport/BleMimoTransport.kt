package com.llmhub.llmhub.mimobot.transport

import android.content.Context
import com.llmhub.llmhub.mimobot.MimoBotIds
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * BLE implementation of [MimoTransport].
 *
 * NOT IMPLEMENTED YET. This is the wiring skeleton so upstream code can be
 * written against a real type. Fill in when you're ready to do the BLE work.
 *
 * TODO(ble):
 *   1. Add to AndroidManifest.xml:
 *        <uses-permission android:name="android.permission.BLUETOOTH_SCAN"
 *            android:usesPermissionFlags="neverForLocation" />
 *        <uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
 *      (and runtime-request them on Android 12+).
 *   2. Scan with a [ScanFilter] on [MimoBotIds.SERVICE] (not name — name filters
 *      are unreliable on newer Android versions).
 *   3. Open GATT, request MTU 247, discover services.
 *   4. Enable notifications on AUDIO_UP + CONTROL characteristics.
 *   5. Strip the 2-byte LE seq prefix from incoming AUDIO_UP payloads before
 *      emitting to [audioUp]; prepend seq on outgoing [sendAudioDown].
 *   6. Use WRITE_WITHOUT_RESPONSE for AUDIO_DOWN; respect the `onCharacteristicWrite`
 *      backpressure callback so you don't overrun the BLE controller.
 *   7. Emit CONTROL characteristic payloads as UTF-8 strings on [control].
 */
class BleMimoTransport(private val context: Context) : MimoTransport {

    private val _state = MutableStateFlow(MimoTransport.State.DISCONNECTED)
    override val state: StateFlow<MimoTransport.State> = _state.asStateFlow()

    private val _audioUp = MutableSharedFlow<ByteArray>(extraBufferCapacity = 64)
    override val audioUp: SharedFlow<ByteArray> = _audioUp.asSharedFlow()

    private val _control = MutableSharedFlow<String>(extraBufferCapacity = 16)
    override val control: SharedFlow<String> = _control.asSharedFlow()

    override suspend fun connect() {
        _state.value = MimoTransport.State.SCANNING
        TODO("BLE scan + GATT connect — see TODO(ble) in class header")
    }

    override suspend fun sendAudioDown(opusFrame: ByteArray) {
        TODO("GATT write-without-response to AUDIO_DOWN with 2-byte seq prefix")
    }

    override suspend fun sendControl(json: String) {
        TODO("GATT write to CONTROL with UTF-8 payload")
    }

    override suspend fun disconnect() {
        _state.value = MimoTransport.State.DISCONNECTED
        // TODO: close BluetoothGatt, unregister callbacks.
    }
}
