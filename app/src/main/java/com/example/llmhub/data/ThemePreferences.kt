package com.llmhub.llmhub.data

import android.content.Context
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
            preferences[WEB_SEARCH_KEY] ?: true // Default to enabled
        }

    val appLanguage: Flow<String?> = context.dataStore.data
        .map { preferences ->
            preferences[LANGUAGE_KEY] // null means system default
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
}
