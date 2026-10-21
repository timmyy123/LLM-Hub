package com.llmhub.llmhub

import android.content.Context
import android.content.res.Configuration
import android.os.Build
import android.os.Bundle
import android.view.animation.PathInterpolator
import androidx.activity.ComponentActivity
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.activity.compose.setContent
import androidx.core.view.WindowCompat
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.Surface
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.LayoutDirection
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.compose.rememberNavController
import kotlinx.coroutines.flow.first
import com.llmhub.llmhub.data.ThemeMode
import com.llmhub.llmhub.navigation.LlmHubNavigation
import com.llmhub.llmhub.ui.theme.LlmHubTheme
import com.llmhub.llmhub.viewmodels.ChatViewModelFactory
import com.llmhub.llmhub.viewmodels.ThemeViewModel
import com.llmhub.llmhub.utils.LocaleHelper
import com.llmhub.llmhub.utils.CrashNotice
import com.llmhub.llmhub.debug.ScreenshotHooks

class MainActivity : ComponentActivity() {
    private lateinit var themeViewModel: ThemeViewModel

    /**
     * Dark status/navigation bar icons on a light app theme, light icons on a dark one. LlmHubTheme
     * sets this on composition, but on Android 12+ AndroidX's splash remove() re-applies the XML
     * theme's windowLightStatusBar (unset = light icons) afterwards, which left white icons on the
     * light theme's white top bar (only the battery outline visible).
     */
    private fun applySystemBarAppearance() {
        if (!::themeViewModel.isInitialized) return
        val dark = when (themeViewModel.themeMode.value) {
            ThemeMode.LIGHT -> false
            ThemeMode.DARK -> true
            ThemeMode.SYSTEM -> (resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES
        }
        WindowCompat.getInsetsController(window, window.decorView).apply {
            isAppearanceLightStatusBars = !dark
            isAppearanceLightNavigationBars = !dark
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        // Launch splash (Theme.LlmHub.Starting): ease the icon forward and fade it out while the
        // splash background dissolves into the app, matching the iOS launch animation.
        installSplashScreen().setOnExitAnimationListener { splash ->
            val easing = PathInterpolator(0.4f, 0f, 0.2f, 1f)
            // Some Android 16 builds (Galaxy S23/S26 Ultra, A17, Lenovo tablets) hand over a splash
            // without an icon view, and AndroidX's iconView then throws a NullPointerException.
            runCatching { splash.iconView }.getOrNull()?.animate()
                ?.scaleX(1.12f)?.scaleY(1.12f)?.alpha(0f)
                ?.setDuration(320L)?.setInterpolator(easing)
                ?.start()
            splash.view.animate()
                .alpha(0f)
                .setStartDelay(120L).setDuration(380L).setInterpolator(easing)
                .withEndAction {
                    splash.remove()
                    applySystemBarAppearance()
                }
                .start()
        }
        super.onCreate(savedInstanceState)

        val app = application as LlmHubApplication
        val chatRepository = app.chatRepository
        val chatViewModelFactory = ChatViewModelFactory(app, chatRepository, this)
        val showCrashNotice = CrashNotice.consumeOnLaunch(this)

        // Initialize ThemeViewModel
        themeViewModel = ThemeViewModel(this)

        // Edge-to-edge without Window.setStatusBarColor/setNavigationBarColor (deprecated in
        // Android 15, and still called by AndroidX's enableEdgeToEdge helpers): Android 15+ draws
        // edge-to-edge by itself; below that Theme.LlmHub makes the bars transparent and the
        // window stops fitting system windows. Compose screens handle the insets.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.VANILLA_ICE_CREAM) {
            WindowCompat.setDecorFitsSystemWindows(window, false)
        }
        setContent {
            val currentThemeMode by themeViewModel.themeMode.collectAsState()
            val currentLanguage by themeViewModel.appLanguage.collectAsState()

            // Apply locale (updates Context resources + AppCompat locales)
            LocaleHelper.setLocale(this@MainActivity, currentLanguage)

            // RTL locales: Persian (fa), Arabic (ar), Hebrew (he/iw)
            val isRtl = currentLanguage in setOf("fa", "ar", "he", "iw")
            val layoutDirection = if (isRtl) LayoutDirection.Rtl else LayoutDirection.Ltr

            LlmHubTheme(themeMode = currentThemeMode) {
                // Debug builds wrap the content in ScreenshotHooks' live-locale provider (used only by
                // the Play Store screenshot test); release builds keep the plain RTL provider.
                ScreenshotHooks.LocaleOverrideProvider(this@MainActivity, layoutDirection) {
                    var crashNoticeVisible by remember { mutableStateOf(showCrashNotice) }
                    Surface(
                        modifier = Modifier.fillMaxSize(),
                        color = MaterialTheme.colorScheme.background
                    ) {
                        val navController = rememberNavController()
                        LlmHubNavigation(
                            navController = navController,
                            chatViewModelFactory = chatViewModelFactory,
                            themeViewModel = themeViewModel
                        )
                    }
                    if (crashNoticeVisible) {
                        AlertDialog(
                            onDismissRequest = { crashNoticeVisible = false },
                            title = { Text(stringResource(R.string.crash_recovery_title)) },
                            text = { Text(stringResource(R.string.crash_recovery_message)) },
                            confirmButton = {
                                TextButton(onClick = { crashNoticeVisible = false }) {
                                    Text(stringResource(R.string.ok))
                                }
                            },
                        )
                    }
                } // CompositionLocalProvider
            }
        }
    }
    
    override fun attachBaseContext(newBase: Context) {
        // Get the saved language preference and apply locale configuration
        val themePrefs = com.llmhub.llmhub.data.ThemePreferences(newBase)
        val savedLanguage = try {
            // Try to get the saved language synchronously
            kotlinx.coroutines.runBlocking {
                themePrefs.appLanguage.first()
            }
        } catch (e: Exception) {
            null // Fall back to system default if we can't read preferences
        }
        
        super.attachBaseContext(LocaleHelper.setLocale(newBase, savedLanguage))
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level >= TRIM_MEMORY_BACKGROUND || level >= TRIM_MEMORY_MODERATE || level >= TRIM_MEMORY_RUNNING_CRITICAL) {
            System.gc()
        }
    }
}
