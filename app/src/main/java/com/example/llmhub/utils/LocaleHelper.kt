package com.llmhub.llmhub.utils

import android.content.Context
import android.content.res.Configuration
import android.os.Build
import android.os.LocaleList
import java.util.Locale

object LocaleHelper {
    
    // Supported locales in the app
    private val supportedLocales = listOf(
        "en", // English (default)
        "es", // Spanish
        "pt", // Portuguese
        "de", // German
        "fr", // French
        "ru"  // Russian
    )
    
    /**
     * Set the app locale based on user preference or system default
     * @param context Application context
     * @param languageCode Language code (null for system default)
     * @return Updated context with applied locale
     */
    fun setLocale(context: Context, languageCode: String? = null): Context {
        val locale = getAppropriateLocale(languageCode)
        return updateResources(context, locale)
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
     * Get the current system locale
     */
    private fun getSystemLocale(): Locale {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            LocaleList.getDefault()[0]
        } else {
            @Suppress("DEPRECATION")
            Locale.getDefault()
        }
    }
    
    /**
     * Update context resources with the specified locale
     */
    private fun updateResources(context: Context, locale: Locale): Context {
        Locale.setDefault(locale)
        
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
        return supportedLocales.map { code ->
            val locale = Locale(code)
            val displayName = locale.getDisplayLanguage(locale).replaceFirstChar { 
                if (it.isLowerCase()) it.titlecase(locale) else it.toString() 
            }
            code to displayName
        }
    }
}
