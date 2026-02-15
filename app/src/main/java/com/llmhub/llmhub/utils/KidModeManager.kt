package com.llmhub.llmhub.utils

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * Manages the state and security of "Kid Mode".
 * Stores a PIN and an enabled flag in EncryptedSharedPreferences for security.
 */
class KidModeManager(context: Context) {

    private val prefs: SharedPreferences

    init {
        // Use EncryptedSharedPreferences to securely store the PIN
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()

        prefs = EncryptedSharedPreferences.create(
            context,
            "kid_mode_prefs",
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }

    private val _isKidModeEnabled = MutableStateFlow(prefs.getBoolean(KEY_IS_ENABLED, false))
    val isKidModeEnabled: StateFlow<Boolean> = _isKidModeEnabled.asStateFlow()

    fun enableKidMode(pin: String) {
        if (pin.length != 4 || !pin.all { it.isDigit() }) {
            throw IllegalArgumentException("PIN must be 4 digits")
        }
        prefs.edit().apply {
            putBoolean(KEY_IS_ENABLED, true)
            putString(KEY_PIN, pin) // In a real app, hash this. EncryptedSharedPreferences adds a layer of security.
            apply()
        }
        _isKidModeEnabled.value = true
        Log.d("KidModeManager", "Kid Mode Enabled")
    }

    fun disableKidMode(pin: String): Boolean {
        val storedPin = prefs.getString(KEY_PIN, "")
        if (storedPin == pin) {
            prefs.edit().apply {
                putBoolean(KEY_IS_ENABLED, false)
                remove(KEY_PIN) // Clear the PIN so it must be reset next time
                apply()
            }
            _isKidModeEnabled.value = false
            Log.d("KidModeManager", "Kid Mode Disabled")
            return true
        }
        Log.w("KidModeManager", "Kid Mode Disable Failed: Incorrect PIN")
        return false
    }

    fun verifyPin(pin: String): Boolean {
        val storedPin = prefs.getString(KEY_PIN, "")
        return storedPin == pin
    }

    companion object {
        private const val KEY_IS_ENABLED = "is_enabled"
        private const val KEY_PIN = "pin"
        
        // The system prompt to inject when Kid Mode is enabled
        const val SYSTEM_INSTRUCTION = "You are performing this task for a 14 year old. You must refuse if this is age-inappropriate, and mention that the user should ask a trusted adult instead of an AI model."
    }
}
