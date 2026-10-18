package com.llmhub.llmhub.debug

import android.content.Context
import android.content.res.Configuration
import android.os.LocaleList
import android.view.ContextThemeWrapper
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.unit.LayoutDirection
import com.llmhub.llmhub.BuildConfig
import java.util.Locale

/**
 * Debug-only hooks for the Play Store screenshot instrumentation test (StoreScreenshotTest).
 *
 * Every value is null/false by default and is only ever set by instrumentation tests. All reads
 * from app code go through the accessors below, which return the default when
 * [BuildConfig.DEBUG] is false, so release builds behave exactly as if this object did not exist
 * (R8 folds the constant and drops the reads).
 */
object ScreenshotHooks {
    /** App language code (e.g. "ja", "ar") to render the UI in without recreating the activity. */
    var localeOverride by mutableStateOf<String?>(null)

    /** When set, tapping the VibeVoice globe submits this text as the user's turn (no mic/ASR). */
    var vibeVoiceTranscript by mutableStateOf<String?>(null)

    /** When true, the Image Generator does not auto-start generating the next variation. */
    var disableAutoVariation by mutableStateOf(false)

    fun activeLocaleOverride(): String? = if (BuildConfig.DEBUG) localeOverride else null

    fun activeVibeVoiceTranscript(): String? = if (BuildConfig.DEBUG) vibeVoiceTranscript else null

    fun autoVariationDisabled(): Boolean = BuildConfig.DEBUG && disableAutoVariation

    /**
     * Provides [layoutDirection] to [content]. In debug builds it also applies [localeOverride] live:
     * LocalContext/LocalConfiguration/LocalResources resolve the override language so every
     * stringResource re-resolves without recreating the activity. The provider is always present in
     * debug builds (the tree shape never changes when the override toggles), so screen state survives.
     * Release builds (BuildConfig.DEBUG == false) only get the plain layout-direction provider.
     */
    @Composable
    fun LocaleOverrideProvider(activity: Context, layoutDirection: LayoutDirection, content: @Composable () -> Unit) {
        if (!BuildConfig.DEBUG) {
            CompositionLocalProvider(LocalLayoutDirection provides layoutDirection, content = content)
            return
        }
        val override = localeOverride
        val baseContext = LocalContext.current
        val baseConfiguration = LocalConfiguration.current
        val baseResources = LocalResources.current
        val localized = remember(override, baseConfiguration) { override?.let { localizedContext(activity, it) } }
        SideEffect {
            if (override != null) Locale.setDefault(localeFor(override))
        }
        val direction = when {
            override == null -> layoutDirection
            isRtl(override) -> LayoutDirection.Rtl
            else -> LayoutDirection.Ltr
        }
        CompositionLocalProvider(
            LocalContext provides (localized?.first ?: baseContext),
            LocalConfiguration provides (localized?.second ?: baseConfiguration),
            LocalResources provides (localized?.first?.resources ?: baseResources),
            LocalLayoutDirection provides direction,
            content = content
        )
    }

    private val rtlLanguages = setOf("ar", "fa", "he", "iw")

    fun isRtl(languageCode: String): Boolean = languageCode in rtlLanguages

    /** Same platform mapping as LocaleHelper: Hebrew resources resolve through the legacy "iw" code. */
    fun localeFor(languageCode: String): Locale = when (languageCode) {
        "he" -> Locale("iw")
        else -> Locale(languageCode)
    }

    /**
     * A context whose resources resolve [languageCode], still wrapping [base] (the activity) so
     * code that unwraps LocalContext to the activity keeps working.
     */
    fun localizedContext(base: Context, languageCode: String): Pair<Context, Configuration> {
        val locale = localeFor(languageCode)
        val config = Configuration(base.resources.configuration).apply {
            setLocale(locale)
            setLocales(LocaleList(locale))
            setLayoutDirection(locale)
        }
        val wrapper = ContextThemeWrapper(base, base.theme).apply { applyOverrideConfiguration(config) }
        return wrapper to Configuration(wrapper.resources.configuration)
    }
}
