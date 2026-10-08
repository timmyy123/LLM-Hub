package com.llmhub.llmhub.data

import android.content.Context
import android.os.Build
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

enum class ThemeMode {
    LIGHT,
    DARK,
    SYSTEM
}

class ThemePreferences(private val context: Context) {
    companion object {
        private val THEME_KEY = stringPreferencesKey("theme_mode")
        private val WEB_SEARCH_KEY = booleanPreferencesKey("web_search_enabled")
        private val LANGUAGE_KEY = stringPreferencesKey("app_language")
        private val EMBEDDING_ENABLED_KEY = booleanPreferencesKey("embedding_enabled")
        private val SELECTED_EMBEDDING_MODEL_KEY = stringPreferencesKey("selected_embedding_model")
        private val MEMORY_ENABLED_KEY = booleanPreferencesKey("memory_enabled")
        private val AUTO_READOUT_ENABLED_KEY = booleanPreferencesKey("auto_readout_enabled")
        private val IS_PREMIUM_KEY = booleanPreferencesKey("is_premium")
        private val GITHUB_STARS_KEY = androidx.datastore.preferences.core.intPreferencesKey("github_stars")
        private val POCKET_VOICE_KEY = stringPreferencesKey("pocket_voice")
        private val SUPERTONIC_VOICE_KEY = stringPreferencesKey("supertonic_voice")
        private val SUPERTONIC_LANGUAGE_KEY = stringPreferencesKey("supertonic_language")
        private val SELECTED_TTS_MODEL_KEY = stringPreferencesKey("selected_tts_model")
        private val SELECTED_TTS_DEVICE_KEY = stringPreferencesKey("selected_tts_device")
        private val SELECTED_TTS_VOICE_KEY = stringPreferencesKey("selected_tts_voice")
        private val GGUF_USE_VULKAN_KEY = booleanPreferencesKey("gguf_use_vulkan")
    }

    val themeMode: Flow<ThemeMode> = context.dataStore.data
        .map { preferences ->
            when (preferences[THEME_KEY]) {
                "LIGHT" -> ThemeMode.LIGHT
                "DARK" -> ThemeMode.DARK
                "SYSTEM" -> ThemeMode.SYSTEM
                else -> ThemeMode.SYSTEM // Default to system
            }
        }

    val webSearchEnabled: Flow<Boolean> = context.dataStore.data
        .map { preferences ->
            preferences[WEB_SEARCH_KEY] ?: false // Default to disabled
        }

    val ggufUseVulkan: Flow<Boolean> = context.dataStore.data
        .map { preferences -> preferences[GGUF_USE_VULKAN_KEY] ?: defaultGgufUseVulkan() }

    fun defaultGgufUseVulkan(): Boolean = !isSnapdragonChip()

    private fun isSnapdragonChip(): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
            Build.SOC_MANUFACTURER.contains("qualcomm", ignoreCase = true)
        ) return true
        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else ""
        return listOf(soc, Build.HARDWARE, Build.BOARD).any { identifier ->
            identifier.startsWith("SM", ignoreCase = true) ||
                identifier.startsWith("SDM", ignoreCase = true) ||
                identifier.startsWith("MSM", ignoreCase = true) ||
                identifier.startsWith("QCS", ignoreCase = true) ||
                identifier.startsWith("QCM", ignoreCase = true) ||
                identifier.contains("qcom", ignoreCase = true) ||
                identifier.contains("qualcomm", ignoreCase = true)
        }
    }

    suspend fun setGgufUseVulkan(enabled: Boolean) {
        context.dataStore.edit { preferences -> preferences[GGUF_USE_VULKAN_KEY] = enabled }
    }

    val appLanguage: Flow<String?> = context.dataStore.data
        .map { preferences ->
            preferences[LANGUAGE_KEY] // null means system default
        }

    val embeddingEnabled: Flow<Boolean> = context.dataStore.data
        .map { preferences ->
            preferences[EMBEDDING_ENABLED_KEY] ?: false
        }

    val memoryEnabled: Flow<Boolean> = context.dataStore.data
        .map { preferences ->
            preferences[MEMORY_ENABLED_KEY] ?: false
        }

    val selectedEmbeddingModel: Flow<String?> = context.dataStore.data
        .map { preferences ->
            preferences[SELECTED_EMBEDDING_MODEL_KEY]
        }

    val autoReadoutEnabled: Flow<Boolean> = context.dataStore.data
        .map { preferences ->
            preferences[AUTO_READOUT_ENABLED_KEY] ?: false // Default to disabled
        }

    val isPremium: Flow<Boolean> = context.dataStore.data
        .map { preferences ->
            preferences[IS_PREMIUM_KEY] ?: false
        }

    suspend fun setThemeMode(themeMode: ThemeMode) {
        context.dataStore.edit { preferences ->
            preferences[THEME_KEY] = themeMode.name
        }
    }
    
    suspend fun setWebSearchEnabled(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[WEB_SEARCH_KEY] = enabled
        }
    }
    
    suspend fun setAppLanguage(languageCode: String?) {
        context.dataStore.edit { preferences ->
            if (languageCode != null) {
                preferences[LANGUAGE_KEY] = languageCode
            } else {
                preferences.remove(LANGUAGE_KEY)
            }
        }
    }

    suspend fun setEmbeddingEnabled(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[EMBEDDING_ENABLED_KEY] = enabled
        }
    }

    suspend fun setMemoryEnabled(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[MEMORY_ENABLED_KEY] = enabled
        }
    }

    suspend fun setSelectedEmbeddingModel(modelName: String?) {
        context.dataStore.edit { preferences ->
            if (modelName != null) {
                preferences[SELECTED_EMBEDDING_MODEL_KEY] = modelName
            } else {
                preferences.remove(SELECTED_EMBEDDING_MODEL_KEY)
            }
        }
    }

    suspend fun setAutoReadoutEnabled(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[AUTO_READOUT_ENABLED_KEY] = enabled
        }
    }

    suspend fun setIsPremium(premium: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[IS_PREMIUM_KEY] = premium
        }
    }

    val githubStars: Flow<Int> = context.dataStore.data
        .map { preferences ->
            preferences[GITHUB_STARS_KEY] ?: 0 // Default to 0
        }

    suspend fun setGithubStars(stars: Int) {
        context.dataStore.edit { preferences ->
            preferences[GITHUB_STARS_KEY] = stars
        }
    }

    val pocketVoice: Flow<String> = context.dataStore.data.map { it[POCKET_VOICE_KEY] ?: "" }
    suspend fun setPocketVoice(voice: String) { context.dataStore.edit { it[POCKET_VOICE_KEY] = voice } }
    val supertonicVoice: Flow<String> = context.dataStore.data.map { it[SUPERTONIC_VOICE_KEY]?.takeIf { key -> SupertonicModel.isVoiceKey(key) && (key in SupertonicModel.voices || SupertonicModel.voiceFile(context, key) != null) } ?: "M1" }
    val supertonicLanguage: Flow<String> = context.dataStore.data.map { it[SUPERTONIC_LANGUAGE_KEY] ?: "" }
    suspend fun setSupertonicVoice(voice: String) {
        require(SupertonicModel.isVoiceKey(voice))
        context.dataStore.edit { it[SUPERTONIC_VOICE_KEY] = voice }
    }
    suspend fun setSupertonicLanguage(language: String) {
        require(language.isEmpty() || language in SupertonicModel.languages)
        context.dataStore.edit { it[SUPERTONIC_LANGUAGE_KEY] = language }
    }

    val selectedTtsModel: Flow<String?> = context.dataStore.data
        .map { preferences ->
            preferences[SELECTED_TTS_MODEL_KEY]
        }

    val selectedTtsDevice: Flow<String> = context.dataStore.data
        .map { preferences ->
            preferences[SELECTED_TTS_DEVICE_KEY] ?: "gpu" // Default to GPU
        }

    val selectedTtsVoice: Flow<String> = context.dataStore.data
        .map { preferences ->
            preferences[SELECTED_TTS_VOICE_KEY] ?: "af_sky" // Default to English Sky
        }

    suspend fun setSelectedTtsModel(modelName: String?) {
        context.dataStore.edit { preferences ->
            if (modelName != null) {
                preferences[SELECTED_TTS_MODEL_KEY] = modelName
            } else {
                preferences.remove(SELECTED_TTS_MODEL_KEY)
            }
        }
    }

    suspend fun setSelectedTtsDevice(device: String) {
        context.dataStore.edit { preferences ->
            preferences[SELECTED_TTS_DEVICE_KEY] = device
        }
    }

    suspend fun setSelectedTtsVoice(voice: String) {
        context.dataStore.edit { preferences ->
            preferences[SELECTED_TTS_VOICE_KEY] = voice
        }
    }
}
