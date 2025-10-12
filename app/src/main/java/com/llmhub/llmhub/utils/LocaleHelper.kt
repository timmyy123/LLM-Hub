package com.llmhub.llmhub.utils

import android.content.Context
import android.content.res.Configuration
import android.os.Build
import android.os.LocaleList
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.os.LocaleListCompat
import java.util.Locale

object LocaleHelper {
    
    // Supported locales in the app
    private val supportedLocales = listOf(
        "en", // English (default)
        "es", // Spanish
        "pt", // Portuguese
        "de", // German
        "fr", // French
        "ru", // Russian
        "it", // Italian
        "tr", // Turkish
        "pl", // Polish
        "ar", // Arabic
        "ja", // Japanese
        "id", // Indonesian (modern ISO 639-1 code)
        "in"  // Indonesian (legacy Android code for compatibility)
    )
    
    /**
     * Set the app locale based on user preference or system default
     * @param context Application context
     * @param languageCode Language code (null for system default)
     * @return Updated context with applied locale
     */
    fun setLocale(context: Context, languageCode: String? = null): Context {
        val locale = getAppropriateLocale(languageCode)
        Locale.setDefault(locale)
        applyAppCompatLocales(languageCode)
        return updateResources(context, locale)
    }
    
    /**
     * Apply locale changes to current context
     * This method can be called to update the locale for an existing context
     */
    fun applyLocale(context: Context, languageCode: String?) {
        val locale = getAppropriateLocale(languageCode)
        Locale.setDefault(locale)
        applyAppCompatLocales(languageCode)
        updateResourcesInPlace(context, locale)
    }
    
    /**
     * Also set per-app locales via AppCompat to ensure consistency across components
     */
    private fun applyAppCompatLocales(languageCode: String?) {
        val locales = if (languageCode.isNullOrEmpty()) {
            LocaleListCompat.getEmptyLocaleList()
        } else {
            // Handle Indonesian language code properly
            // Android resources use "id" but Locale internally uses "in" for legacy reasons
            // We need to use "in" for AppCompat locale setting for proper resource resolution
            val normalizedCode = when (languageCode) {
                "id" -> "in" // Indonesian: use legacy "in" for AppCompat
                else -> languageCode
            }
            LocaleListCompat.forLanguageTags(normalizedCode)
        }
        AppCompatDelegate.setApplicationLocales(locales)
    }
    
    /**
     * Get the appropriate locale based on preference or system default
     * Falls back to English if the system/preference locale is not supported
     */
    private fun getAppropriateLocale(languageCode: String?): Locale {
        return when {
            // Use explicit language code if provided and supported
            !languageCode.isNullOrEmpty() && supportedLocales.contains(languageCode) -> {
                Locale(languageCode)
            }
            // Try to use system locale if supported
            else -> {
                val systemLocale = getSystemLocale()
                val systemLanguage = systemLocale.language
                
                if (supportedLocales.contains(systemLanguage)) {
                    Locale(systemLanguage)
                } else {
                    // Fall back to English
                    Locale("en")
                }
            }
        }
    }
    
    /**
     * Get the current system locale from device configuration
     * This gets the device's actual system locale, not the app's current locale
     */
    private fun getSystemLocale(): Locale {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                // Get the system's default locale list, not the app's current one
                val systemConfig = android.content.res.Resources.getSystem().configuration
                systemConfig.locales[0]
            } else {
                @Suppress("DEPRECATION")
                // For older versions, get from system resources
                android.content.res.Resources.getSystem().configuration.locale
            }
        } catch (e: Exception) {
            // Fallback to English if system locale detection fails
            Locale("en")
        }
    }
    
    /**
     * Update context resources with the specified locale
     */
    private fun updateResources(context: Context, locale: Locale): Context {
        val config = Configuration(context.resources.configuration)
        config.setLocale(locale)
        
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            config.setLocales(LocaleList(locale))
            context.createConfigurationContext(config)
        } else {
            @Suppress("DEPRECATION")
            config.locale = locale
            context.createConfigurationContext(config)
        }
    }
    
    /**
     * Update resources in place for existing context
     */
    private fun updateResourcesInPlace(context: Context, locale: Locale) {
        val config = Configuration(context.resources.configuration)
        config.setLocale(locale)
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            config.setLocales(LocaleList(locale))
        } else {
            @Suppress("DEPRECATION")
            config.locale = locale
        }
        
        @Suppress("DEPRECATION")
        context.resources.updateConfiguration(config, context.resources.displayMetrics)
    }
    
    /**
     * Get the current app locale
     */
    fun getCurrentLocale(context: Context): Locale {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            context.resources.configuration.locales[0]
        } else {
            @Suppress("DEPRECATION")
            context.resources.configuration.locale
        }
    }
    
    /**
     * Get the language code from a locale
     */
    fun getLanguageCode(locale: Locale): String {
        return locale.language
    }
    
    /**
     * Check if a language code is supported
     */
    fun isLanguageSupported(languageCode: String): Boolean {
        return supportedLocales.contains(languageCode)
    }
    
    /**
     * Get list of supported locales with their display names
     */
    fun getSupportedLanguages(context: Context): List<Pair<String, String>> {
        return supportedLocales
            .filter { it != "in" } // Filter out legacy "in" code to avoid duplicate Indonesian entry
            .map { code ->
                val locale = Locale(code)
                val displayName = locale.getDisplayLanguage(locale).replaceFirstChar { 
                    if (it.isLowerCase()) it.titlecase(locale) else it.toString() 
                }
                code to displayName
            }
    }
    
    /**
     * Normalize language code to handle legacy mappings
     * Indonesian: both "id" and "in" map to "id" for user preference storage
     */
    fun normalizeLanguageCode(languageCode: String): String {
        return when (languageCode) {
            "in" -> "id" // Normalize legacy Indonesian code to modern code
            else -> languageCode
        }
    }
}
