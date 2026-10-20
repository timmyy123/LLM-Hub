package com.llmhub.llmhub.screenshots

import android.Manifest
import android.app.Activity
import android.app.Instrumentation
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.SemanticsNodeInteraction
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasContentDescription
import androidx.compose.ui.test.hasScrollToNodeAction
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.isEnabled
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.performTextInput
import androidx.core.content.FileProvider
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.intent.Intents
import androidx.test.espresso.intent.Intents.intending
import androidx.test.espresso.intent.matcher.IntentMatchers.hasAction
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import com.llmhub.llmhub.LlmHubApplication
import com.llmhub.llmhub.MainActivity
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.debug.ScreenshotHooks
import com.llmhub.llmhub.imagegen.ModelType
import com.llmhub.llmhub.imagegen.StableDiffusionHelper
import com.llmhub.llmhub.utils.LocaleHelper
import com.llmhub.llmhub.videogen.VideoGenBackend
import com.llmhub.llmhub.videogen.VideoGenEngine
import com.llmhub.llmhub.videogen.VideoGenFormats
import com.llmhub.llmhub.viewmodels.ModelDownloadViewModel
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.FixMethodOrder
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File
import java.util.Locale

/**
 * Play Store screenshots with real on-device results (Device Farm: Galaxy S25 / Galaxy Tab S11).
 *
 * Each feature is run once in English, then the same screen is re-captured in every app language
 * through the debug-only [ScreenshotHooks.localeOverride] (no activity recreation, results stay).
 *
 * Opt-in: `am instrument -e store_screenshots true -e class com.llmhub.llmhub.screenshots.StoreScreenshotTest#<method>`.
 * Output: `<external files>/store_screenshots/<lang>/<NN>_<feature>.png`, failures as
 * `store_screenshots/debug_<feature>_failure.png`. Logcat tag `StoreScreenshots`, `SHOT_OK <feature> <lang>`.
 *
 * Optional args: `langs` (comma list, default all 22), `llm_model` (default LFM-2.5 1.2B Instruct (Q4_0)),
 * `translate_to` (default ja), `image_steps` (20), `image_seed`, `video_model` (substring), `video_backend`
 * (NPU/GPU/CPU), `video_steps` (8), `video_width`/`video_height`/`video_frames` (GGUF only; 512/320/25),
 * `video_seed` (42).
 *
 * Methods run in name order; s02 (image) saves the image that s05 (upscale) uses.
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class StoreScreenshotTest {

    @get:Rule
    val compose = createEmptyComposeRule()

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val args = InstrumentationRegistry.getArguments()
    private val ctx: Context = instrumentation.targetContext
    private val device: UiDevice = UiDevice.getInstance(instrumentation)

    private var scenario: ActivityScenario<MainActivity>? = null
    private lateinit var activity: MainActivity

    private val outRoot: File by lazy {
        File(ctx.getExternalFilesDir(null), "store_screenshots").apply { mkdirs() }
    }
    private val languages: List<String> by lazy {
        args.getString("langs")?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() } ?: ALL_LANGUAGES
    }
    private val llmName: String get() = args.getString("llm_model") ?: DEFAULT_LLM

    @Before
    fun setUp() {
        assumeTrue("Enable explicitly with -e store_screenshots true", args.getString("store_screenshots") == "true")
        runBlocking {
            ThemePreferences(ctx).setIsPremium(true)
            ThemePreferences(ctx).setAppLanguage("en")
        }
        instrumentation.runOnMainSync { LocaleHelper.applyLocale(ctx.applicationContext, "en") }
        Locale.setDefault(Locale.ENGLISH)
        val permissions = buildList {
            add(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
        }
        for (permission in permissions) {
            runCatching { instrumentation.uiAutomation.grantRuntimePermission(ctx.packageName, permission) }
                .onFailure { Log.w(TAG, "grant $permission failed: ${it.message}") }
        }
        runCatching { device.setOrientationPortrait() }.onFailure { Log.w(TAG, "setOrientationPortrait: ${it.message}") }
        runCatching { device.freezeRotation() }
        instrumentation.runOnMainSync {
            ScreenshotHooks.localeOverride = null
            ScreenshotHooks.vibeVoiceTranscript = null
            ScreenshotHooks.disableAutoVariation = true
        }
        Log.i(TAG, "device=${Build.MANUFACTURER} ${Build.MODEL} soc=${soc()} qualcommNpu=${DeviceInfo.isQualcommNpuSupported()} " +
            "display=${device.displayWidth}x${device.displayHeight} langs=$languages")
    }

    @After
    fun tearDown() {
        instrumentation.runOnMainSync {
            ScreenshotHooks.localeOverride = null
            ScreenshotHooks.vibeVoiceTranscript = null
            ScreenshotHooks.disableAutoVariation = false
        }
        Locale.setDefault(Locale.ENGLISH)
        runCatching { scenario?.close() }
        scenario = null
        runCatching { device.unfreezeRotation() }
    }

    // ---------------------------------------------------------------------------------------------
    // Features
    // ---------------------------------------------------------------------------------------------

    @Test
    fun s00_home() {
        runFeature("home") {
            launchApp()
            sleep(2_000) // entrance animations
            captureAllLanguages("00_home", settleMs = 800)
        }
    }

    @Test
    fun s01_vibecode() {
        runFeature("vibecode") {
            val llm = llmName
            val project = File(ctx.filesDir, "vibe_demo").apply { deleteRecursively(); mkdirs() }
            File(project, "index.html").writeText(VIBE_SKELETON)
            ctx.getSharedPreferences("vibe_coder_prefs", Context.MODE_PRIVATE).edit().clear()
                .putString("selected_model_name", llm)
                .putString("selected_backend", "CPU")
                .putString("selected_backend_$llm", "CPU")
                .putBoolean("enable_thinking", false)
                .putBoolean("enable_thinking_$llm", false)
                .putInt("max_tokens_$llm", 8192)
                .putBoolean("codex_enabled", false)
                .putString("last_opened_folder_uri", project.absolutePath)
                .commit()

            launchApp()
            ensureDownloaded(llm, timeoutMin = 25)
            openFeature(R.string.feature_vibe_coder)

            val chatInput = waitForChatInput(str(R.string.vibe_coder_ask_ai_edit))
            // Editor and chat box only: the request is typed but not sent (no generation).
            chatInput.performTextInput(VIBE_PROMPT)
            hideKeyboard()
            sleep(1_500)
            captureAllLanguages("01_vibecode", settleMs = 800)
        }
    }

    @Test
    fun s02_imageGeneration() {
        runFeature("image") {
            val useNpu = DeviceInfo.isQualcommNpuSupported()
            val model = if (useNpu) ModelData.models.first { it.name.startsWith("Absolute Reality (NPU") }
            else ModelData.models.first { it.name == "Absolute Reality (CPU)" }
            Log.i(TAG, "image model=${model.name}")

            launchApp()
            ensureDownloaded(model.name, timeoutMin = 25)

            val sdModels = StableDiffusionHelper(ctx).listModels()
            val wantedType = if (useNpu) ModelType.QNN_NPU else ModelType.MNN_CPU
            val sdModel = sdModels.firstOrNull { it.type == wantedType && it.path.contains(model.name.replace(" ", "_")) }
                ?: sdModels.firstOrNull { it.type == wantedType }
                ?: throw AssertionError("Downloaded SD model not found by StableDiffusionHelper: $sdModels")
            ctx.getSharedPreferences("image_generator_prefs", Context.MODE_PRIVATE).edit()
                .putInt("iterations", intArg("image_steps", 20))
                .putInt("seed", intArg("image_seed", 20251))
                .putInt("image_width", 512)
                .putInt("image_height", 512)
                .putString("last_model_path", sdModel.path)
                .putBoolean("use_gpu", false)
                .commit()

            openFeature(R.string.feature_image_generator)
            val field = waitForNode("prompt field", hasSetTextAction(), 60_000)
            field.performTextInput(IMAGE_PROMPT)
            hideKeyboard()
            clickFirst(hasText(str(R.string.image_generator_generate)) and hasClickAction() and isEnabled(), scroll = true)

            val success = hasText(str(R.string.image_generator_success))
            val generate = hasText(str(R.string.image_generator_generate)) and hasClickAction() and isEnabled()
            waitForResult("image generated", timeoutMs = 25 * MIN, done = { exists(success) }, idle = { exists(generate) })
            sleep(1_500)
            saveUpscaleInput()
            scrollIntoView(hasText(str(R.string.image_generator_save)) and hasClickAction())
            hideKeyboard()
            captureAllLanguages("02_image", settleMs = 800)
        }
    }

    @Test
    fun s03_videoGeneration() {
        runFeature("video") {
            val pick = args.getString("video_model")
            val model = when {
                pick != null -> ModelData.videoGenerationModels.first { pick in it.name }
                else -> ModelData.videoGenerationModels.firstOrNull { it.modelFormat == VideoGenFormats.QNN }
                    ?: ModelData.videoGenerationModels.first { it.name.contains("(Q4_0)") }
            }
            val supported = VideoGenEngine.supportedBackends(model)
            val backend = args.getString("video_backend")?.let { VideoGenBackend.valueOf(it) }?.takeIf { it in supported }
                ?: supported.first()
            Log.i(TAG, "video model=${model.name} backend=$backend fixedShape=${VideoGenEngine.fixedShape(model)}")
            ctx.getSharedPreferences("video_generator_prefs", Context.MODE_PRIVATE).edit()
                .putString("last_model", model.name)
                .putString("backend_${model.name}", backend.name)
                .putInt("steps", intArg("video_steps", 8))
                .putInt("width", intArg("video_width", 512))
                .putInt("height", intArg("video_height", 320))
                .putInt("num_frames", intArg("video_frames", 25))
                .putInt("fps", 24)
                .putLong("seed", intArg("video_seed", 42).toLong())
                .commit()

            launchApp()
            ensureDownloaded(model.name, timeoutMin = 60)
            openFeature(R.string.feature_video_generator)

            loadModelFromSettingsSheet("video", loadTimeoutMs = 15 * MIN)

            val field = waitForNode("prompt field", hasSetTextAction(), 60_000)
            field.performTextInput(args.getString("video_prompt") ?: VIDEO_PROMPT)
            hideKeyboard()
            clickFirst(hasText(str(R.string.video_generator_generate)) and hasClickAction() and isEnabled(), scroll = true)

            val resultTitle = hasText(str(R.string.video_generator_result_title))
            val generate = hasText(str(R.string.video_generator_generate)) and hasClickAction() and isEnabled()
            waitForResult("video generated", timeoutMs = 45 * MIN, done = { exists(resultTitle) }, idle = { exists(generate) })
            scrollIntoView(hasText(str(R.string.video_generator_save)) and hasClickAction())
            hideKeyboard()
            sleep(4_000) // VideoView prepare + first loop
            captureAllLanguages("03_video", settleMs = 1_500)
        }
    }

    @Test
    fun s04_musicGeneration() {
        runFeature("music") {
            val model = ModelData.musicGenerationModels.first { it.name == "SoundGen Quick (LiteRT)" }
            ctx.getSharedPreferences("music_generator_prefs", Context.MODE_PRIVATE).edit()
                .putString("selected_model_name", model.name)
                .commit()

            launchApp()
            ensureDownloaded(model.name, timeoutMin = 20)
            openFeature(R.string.feature_music_generator)

            clickFirst(hasText(MUSIC_PRESET) and hasClickAction())
            hideKeyboard()
            clickFirst(hasText(str(R.string.generate_music)) and hasClickAction() and isEnabled(), scroll = true)

            val generate = hasText(str(R.string.generate_music)) and hasClickAction() and isEnabled()
            waitForResult("music track", timeoutMs = 15 * MIN,
                done = { exists(hasContentDescription("Play") and hasClickAction()) }, idle = { exists(generate) })
            sleep(2_000)
            hideKeyboard()
            captureAllLanguages("04_music", settleMs = 800)
        }
    }

    @Test
    fun s05_imageUpscale() {
        runFeature("upscale") {
            val model = ModelData.upscalerModels.first { it.name == "Real-ESRGAN General x4v3 (GPU/CPU)" }
            launchApp()
            ensureDownloaded(model.name, timeoutMin = 10)
            val safeName = model.name.replace(Regex("[^a-zA-Z0-9_\\-]"), "_")
            val modelFile = File(ctx.filesDir, "upscaler_models/$safeName/upscaler.tflite")
            check(modelFile.isFile) { "Upscaler not found at $modelFile" }
            ctx.getSharedPreferences("image_upscale_prefs", Context.MODE_PRIVATE).edit()
                .putString("last_model_path", modelFile.absolutePath)
                .commit()

            val input = upscaleInputFile()
            val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", input)
            Intents.init()
            try {
                intending(hasAction(Intent.ACTION_GET_CONTENT)).respondWith(
                    Instrumentation.ActivityResult(
                        Activity.RESULT_OK,
                        Intent().setData(uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    )
                )
                openFeature(R.string.feature_image_upscale)
                clickFirst(hasText(str(R.string.image_upscale_select_image)) and hasClickAction())
                val upscale = hasText(str(R.string.image_upscale_button)) and hasClickAction() and isEnabled()
                waitUntil("input image loaded", timeoutMs = 30_000) { exists(upscale) }
                clickFirst(upscale)
                waitUntil("image upscaled", timeoutMs = 10 * MIN) {
                    exists(hasText(str(R.string.image_upscale_save)) and hasClickAction())
                }
            } finally {
                Intents.release()
            }
            sleep(1_500)
            captureAllLanguages("05_upscale", settleMs = 800)
        }
    }

    @Test
    fun s06_vibeVoice() {
        runFeature("vibevoice") {
            val llm = llmName
            ctx.getSharedPreferences("vibevoice_prefs", Context.MODE_PRIVATE).edit()
                .putString("selected_model_name", llm)
                .putString("selected_voice_model_name", ASR_MODEL)
                .putString("selected_backend", "CPU")
                .remove("selected_npu_device_id")
                .putString("selected_asr_backend", "CPU")
                .commit()

            launchApp()
            // Both must be on disk before the screen opens: hasRequiredModels is computed in the VM init.
            ensureDownloaded(llm, timeoutMin = 20)
            ensureDownloaded(ASR_MODEL, timeoutMin = 10)
            instrumentation.runOnMainSync { ScreenshotHooks.vibeVoiceTranscript = VIBEVOICE_TRANSCRIPT }
            openFeature(R.string.feature_vibevoice)

            loadModelFromSettingsSheet("vibevoice", loadTimeoutMs = 10 * MIN)

            val globe = hasTestTag("vibevoice_globe")
            waitForNode("globe", globe, 60_000)
            clickFirst(globe)

            // Wait until the streamed reply has stopped changing.
            var last = ""
            var stableSince = System.currentTimeMillis()
            waitUntil("vibevoice reply", timeoutMs = 10 * MIN, pollMs = 2_000) {
                val text = textOf(hasTestTag("vibevoice_reply"))
                val now = System.currentTimeMillis()
                if (text != last) {
                    last = text
                    stableSince = now
                }
                text.length > 20 && now - stableSince >= 10_000
            }
            Log.i(TAG, "vibevoice reply: $last")
            hideKeyboard()
            captureAllLanguages("06_vibevoice", settleMs = 800)
        }
    }

    @Test
    fun s07_translator() {
        runFeature("translator") {
            val llm = llmName
            ctx.getSharedPreferences("translator_prefs", Context.MODE_PRIVATE).edit()
                .putString("selected_model_name", llm)
                .putString("selected_backend", "CPU")
                .putString("source_lang", "en")
                .putString("target_lang", args.getString("translate_to") ?: "ja")
                .putBoolean("auto_detect", false)
                .putBoolean("vision_enabled", false)
                .putBoolean("audio_enabled", false)
                .commit()

            launchApp()
            ensureDownloaded(llm, timeoutMin = 20)
            openFeature(R.string.feature_translator)

            val field = waitForNode("input field", hasSetTextAction(), 60_000)
            field.performTextInput(TRANSLATOR_INPUT)
            hideKeyboard()
            val translate = hasText(str(R.string.translator_translate)) and hasClickAction()
            clickFirst(translate and isEnabled())
            sleep(3_000)
            waitUntil("translation", timeoutMs = 10 * MIN, pollMs = 2_000) {
                exists(translate) && exists(hasContentDescription("Copy"))
            }
            sleep(1_500)
            hideKeyboard()
            captureAllLanguages("07_translator", settleMs = 800)
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------

    private fun runFeature(feature: String, body: () -> Unit) {
        val t0 = System.currentTimeMillis()
        Log.i(TAG, "FEATURE_START $feature")
        try {
            body()
            Log.i(TAG, "FEATURE_OK $feature in ${(System.currentTimeMillis() - t0) / 1000}s")
        } catch (t: Throwable) {
            Log.e(TAG, "FEATURE_FAILED $feature after ${(System.currentTimeMillis() - t0) / 1000}s", t)
            runCatching { instrumentation.runOnMainSync { ScreenshotHooks.localeOverride = null } }
            runCatching { device.takeScreenshot(File(outRoot, "debug_${feature}_failure.png")) }
            runCatching { dumpTree(feature) }
            throw t
        }
    }

    private fun launchApp() {
        scenario = ActivityScenario.launch(MainActivity::class.java)
        scenario!!.onActivity { activity = it }
        sleep(1_500)
        dismissSystemDialogs()
        // In-app crash notice ("App closed unexpectedly" / OK) shown after a previous native crash.
        if (exists(hasText(str(R.string.crash_recovery_title)))) {
            Log.w(TAG, "dismissing crash notice")
            clickFirst(hasText(str(R.string.ok)) and hasClickAction())
        }
        waitUntil("premium unlocked", timeoutMs = 20_000) {
            (activity.application as LlmHubApplication).billingManager.isPremium.value
        }
        waitForNode("home", hasText(str(R.string.home_section_tools)), 30_000)
    }

    private fun dismissSystemDialogs() {
        for (label in listOf("Close app", "OK", "Wait")) {
            val obj = device.findObject(By.text(label).pkg("android")) ?: continue
            Log.w(TAG, "dismissing system dialog button '$label'")
            runCatching { obj.click() }
        }
    }

    private fun str(id: Int): String = activity.getString(id)

    private fun intArg(name: String, default: Int): Int = args.getString(name)?.toIntOrNull() ?: default

    private fun soc(): String =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE

    private fun sleep(ms: Long) = Thread.sleep(ms)

    private var lastLookupErrorLog = 0L

    private fun exists(matcher: SemanticsMatcher): Boolean =
        runCatching { compose.onAllNodes(matcher).fetchSemanticsNodes().isNotEmpty() }
            .onFailure { logLookupError(it) }
            .getOrDefault(false)

    private fun logLookupError(t: Throwable) {
        val now = System.currentTimeMillis()
        if (now - lastLookupErrorLog > 30_000) {
            lastLookupErrorLog = now
            Log.w(TAG, "semantics lookup failed: ${t.javaClass.simpleName}: ${t.message?.take(300)}")
        }
    }

    /** Compose + main-looper idle; never throws (a busy app only delays the capture). */
    private fun idle() {
        runCatching { compose.waitForIdle() }.onFailure { logLookupError(it) }
    }

    private fun textOf(matcher: SemanticsMatcher): String = runCatching {
        compose.onAllNodes(matcher, useUnmergedTree = true).fetchSemanticsNodes().firstOrNull()
            ?.config?.getOrNull(SemanticsProperties.Text)?.joinToString("") { it.text }
    }.getOrNull().orEmpty()

    private fun waitUntil(what: String, timeoutMs: Long, pollMs: Long = 1_000, condition: () -> Boolean) {
        val start = System.currentTimeMillis()
        var lastLog = start
        while (true) {
            if (condition()) return
            val now = System.currentTimeMillis()
            if (now - start > timeoutMs) throw AssertionError("Timed out after ${timeoutMs / 1000}s waiting for $what")
            if (now - lastLog > 60_000) {
                Log.i(TAG, "still waiting for $what (${(now - start) / 1000}s)")
                lastLog = now
            }
            Thread.sleep(pollMs)
        }
    }

    /**
     * Waits for [done]. If the screen is back to its idle state ([idle], e.g. the Generate button is
     * enabled again) for 30 s without a result, the run failed: stop early instead of waiting out the timeout.
     */
    private fun waitForResult(what: String, timeoutMs: Long, done: () -> Boolean, idle: () -> Boolean) {
        val start = System.currentTimeMillis()
        var idleSince = 0L
        waitUntil(what, timeoutMs, pollMs = 2_000) {
            if (done()) return@waitUntil true
            val now = System.currentTimeMillis()
            if (now - start > 15_000 && idle()) {
                if (idleSince == 0L) idleSince = now
                if (now - idleSince > 30_000) throw AssertionError("$what: screen went idle without a result")
            } else {
                idleSince = 0L
            }
            false
        }
    }

    private fun waitForNode(what: String, matcher: SemanticsMatcher, timeoutMs: Long): SemanticsNodeInteraction {
        waitUntil(what, timeoutMs) { exists(matcher) }
        return compose.onAllNodes(matcher).onFirst()
    }

    /** Clicks the first node matching [matcher], retrying flaky taps (recomposition, animations). */
    private fun clickFirst(matcher: SemanticsMatcher, scroll: Boolean = false, timeoutMs: Long = 30_000) {
        val start = System.currentTimeMillis()
        var lastError: Throwable? = null
        while (System.currentTimeMillis() - start < timeoutMs) {
            try {
                val node = compose.onAllNodes(matcher).onFirst()
                if (scroll) runCatching { node.performScrollTo() }
                node.performClick()
                idle()
                return
            } catch (t: Throwable) {
                lastError = t
                Thread.sleep(1_000)
            }
        }
        throw AssertionError("Could not click $matcher", lastError)
    }

    private fun scrollIntoView(matcher: SemanticsMatcher) {
        runCatching { compose.onAllNodes(matcher).onFirst().performScrollTo() }
            .onFailure { Log.w(TAG, "scrollIntoView failed: ${it.message}") }
        idle()
    }

    /** Opens a Home tool card by its title, scrolling the Home list when needed. */
    private fun openFeature(titleRes: Int) {
        val title = str(titleRes)
        val card = hasText(title) and hasClickAction()
        waitUntil("home list", timeoutMs = 30_000) { exists(hasScrollToNodeAction()) }
        if (!exists(card)) {
            runCatching { compose.onAllNodes(hasScrollToNodeAction()).onFirst().performScrollToNode(hasText(title)) }
        }
        runCatching { compose.onAllNodes(card).onFirst().performScrollTo() }
        clickFirst(card)
        waitUntil("$title screen", timeoutMs = 30_000) { !exists(hasText(str(R.string.home_section_tools))) }
        sleep(1_500)
    }

    /** Opens the screen's settings sheet (top bar Tune icon), loads the model and closes the sheet. */
    private fun loadModelFromSettingsSheet(feature: String, loadTimeoutMs: Long) {
        val settings = hasContentDescription(str(R.string.feature_settings_title)) and hasClickAction()
        val load = hasText(str(R.string.load_model)) and hasClickAction()
        val reload = hasText(str(R.string.reload_model))
        val loading = hasText(str(R.string.model_loading), substring = true)
        waitUntil("$feature settings button", timeoutMs = 60_000) { exists(settings) }
        clickFirst(settings)
        waitUntil("$feature load button", timeoutMs = 60_000) { exists(load) || exists(reload) }
        if (!exists(reload)) {
            clickFirst(load and isEnabled(), scroll = true)
            sleep(5_000)
            val start = System.currentTimeMillis()
            waitUntil("$feature model loaded", timeoutMs = loadTimeoutMs, pollMs = 2_000) {
                if (exists(reload)) return@waitUntil true
                // Back to an enabled Load button without the loading spinner = the load failed.
                if (System.currentTimeMillis() - start > 20_000 && !exists(loading) && exists(load and isEnabled())) {
                    throw AssertionError("$feature model failed to load")
                }
                false
            }
        }
        sleep(1_000)
        device.pressBack() // close the bottom sheet
        waitUntil("$feature settings sheet closed", timeoutMs = 20_000) { !exists(reload) }
        sleep(1_000)
    }

    /** The VibeCoder chat input: by placeholder, else the empty text field (the editor holds the file). */
    private fun waitForChatInput(hint: String): SemanticsNodeInteraction {
        val byHint = hasSetTextAction() and hasText(hint, substring = true)
        val emptyField = hasSetTextAction() and SemanticsMatcher("empty editable text") {
            it.config.getOrNull(SemanticsProperties.EditableText)?.text?.isEmpty() == true
        }
        // Fallback only once the editor shows the opened file, so the empty field is the chat input.
        val editorWithFile = hasSetTextAction() and hasText("<title>Coffee Shop</title>", substring = true)
        waitUntil("vibecode chat input", timeoutMs = 120_000) {
            exists(byHint) || (exists(editorWithFile) && exists(emptyField))
        }
        sleep(1_000)
        return compose.onAllNodes(if (exists(byHint)) byHint else emptyField).onFirst()
    }

    /** ActivityScenario tracks recreation; keep [activity] pointing at the live instance. */
    private fun refreshActivity() {
        scenario?.onActivity { activity = it }
    }

    private fun hideKeyboard() {
        refreshActivity()
        instrumentation.runOnMainSync {
            runCatching {
                val window = activity.window
                WindowCompat.getInsetsController(window, window.decorView).hide(WindowInsetsCompat.Type.ime())
                activity.currentFocus?.clearFocus()
            }
        }
        idle()
        Thread.sleep(500)
    }

    /**
     * Downloads [name] through the app's own ModelDownloadViewModel (same files/extraction as the
     * Models screen) and waits for it. Interrupted downloads are resumed.
     */
    private fun ensureDownloaded(name: String, timeoutMin: Long) {
        refreshActivity()
        lateinit var vm: ModelDownloadViewModel
        instrumentation.runOnMainSync {
            vm = ViewModelProvider(activity, ViewModelProvider.AndroidViewModelFactory.getInstance(activity.application))[ModelDownloadViewModel::class.java]
        }
        waitUntil("model catalog", timeoutMs = 30_000) { vm.models.value.isNotEmpty() }
        fun current() = vm.models.value.firstOrNull { it.name == name }
            ?: throw AssertionError("Model '$name' is not in the catalog on this device (${soc()})")
        if (current().isDownloaded) {
            Log.i(TAG, "model ready: $name")
            return
        }
        val start = System.currentTimeMillis()
        var attempts = 0
        var lastLog = 0L
        while (true) {
            val model = current()
            if (model.isDownloaded) break
            if (!model.isDownloading && !model.isExtracting) {
                if (attempts >= 8) throw AssertionError("Download of $name failed after $attempts attempts")
                attempts++
                Log.i(TAG, "downloading $name (attempt $attempts, ${model.downloadedBytes / 1_000_000} MB on disk)")
                instrumentation.runOnMainSync { vm.downloadModel(model) }
                Thread.sleep(10_000)
                continue
            }
            val now = System.currentTimeMillis()
            if (now - start > timeoutMin * MIN) throw AssertionError("Download of $name timed out after $timeoutMin min")
            if (now - lastLog > 30_000) {
                Log.i(TAG, "download $name: ${model.downloadedBytes / 1_000_000}/${(model.totalBytes ?: model.sizeBytes) / 1_000_000} MB " +
                    "extracting=${model.isExtracting}")
                lastLog = now
            }
            Thread.sleep(5_000)
        }
        Log.i(TAG, "model downloaded: $name in ${(System.currentTimeMillis() - start) / 1000}s")
    }

    /**
     * Re-captures the current screen in every app language without leaving it, via the live locale
     * override. Files: store_screenshots/<lang>/<name>.png.
     */
    private fun captureAllLanguages(name: String, settleMs: Long) {
        try {
            for (lang in languages) {
                instrumentation.runOnMainSync { ScreenshotHooks.localeOverride = lang }
                idle()
                Thread.sleep(settleMs)
                idle()
                val dir = File(outRoot, lang).apply { mkdirs() }
                val file = File(dir, "$name.png")
                if (!device.takeScreenshot(file)) {
                    Thread.sleep(1_000)
                    check(device.takeScreenshot(file)) { "takeScreenshot failed for $name $lang" }
                }
                Log.i(TAG, "SHOT_OK ${name.substringAfter('_')} $lang ${file.absolutePath}")
            }
        } finally {
            instrumentation.runOnMainSync { ScreenshotHooks.localeOverride = null }
            Locale.setDefault(Locale.ENGLISH)
            idle()
        }
    }

    private val upscaleInput: File get() = File(ctx.filesDir, "store_screenshots_input/upscale_input.png")

    /** Saves the generated image (cropped inside its rounded corners, 512 px) as the upscale input. */
    private fun saveUpscaleInput() {
        runCatching {
            val node = compose.onAllNodes(hasContentDescription("$IMAGE_PROMPT - variation 1")).onFirst()
            val captured = node.captureToImage().asAndroidBitmap()
            val inset = (minOf(captured.width, captured.height) * 0.05f).toInt()
            val side = minOf(captured.width, captured.height) - 2 * inset
            val cropped = Bitmap.createBitmap(captured, inset, inset, side, side)
            val scaled = Bitmap.createScaledBitmap(cropped, 512, 512, true)
            upscaleInput.parentFile?.mkdirs()
            upscaleInput.outputStream().use { scaled.compress(Bitmap.CompressFormat.PNG, 100, it) }
            Log.i(TAG, "saved upscale input ${upscaleInput.absolutePath}")
        }.onFailure { Log.w(TAG, "could not save upscale input: ${it.message}") }
    }

    private fun upscaleInputFile(): File {
        if (upscaleInput.isFile && upscaleInput.length() > 0 && BitmapFactory.decodeFile(upscaleInput.absolutePath) != null) {
            Log.i(TAG, "upscale input: generated image")
            return upscaleInput
        }
        Log.w(TAG, "upscale input: generated image missing, using bundled fallback")
        val fallback = File(ctx.cacheDir, "store_screenshots_input/upscale_fallback.jpg")
        fallback.parentFile?.mkdirs()
        instrumentation.context.assets.open("upscale_fallback.jpg").use { input ->
            fallback.outputStream().use { input.copyTo(it) }
        }
        return fallback
    }

    private fun dumpTree(feature: String) {
        val dump = StringBuilder()
        compose.onAllNodes(SemanticsMatcher("any") { true }).fetchSemanticsNodes().forEach { node ->
            val text = node.config.getOrNull(SemanticsProperties.Text)?.joinToString { it.text }
            val desc = node.config.getOrNull(SemanticsProperties.ContentDescription)?.joinToString()
            val tag = node.config.getOrNull(SemanticsProperties.TestTag)
            if (text != null || desc != null || tag != null) dump.append("text=$text desc=$desc tag=$tag\n")
        }
        File(outRoot, "debug_${feature}_semantics.txt").writeText(dump.toString())
    }

    private companion object {
        const val TAG = "StoreScreenshots"
        const val MIN = 60_000L

        val ALL_LANGUAGES = listOf(
            "en", "es", "pt", "de", "fr", "ru", "it", "tr", "pl", "ar", "ja",
            "id", "ko", "fa", "he", "hi", "uk", "zh", "nl", "da", "th", "vi"
        )

        const val DEFAULT_LLM = "LFM-2.5 1.2B Instruct (Q4_0)"
        const val ASR_MODEL = "Whisper Tiny (English-only)"

        const val VIBE_PROMPT = "Build a beautiful animated landing page for a coffee shop with a hero, " +
            "menu cards and a contact form"
        const val VIBE_SKELETON = "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n" +
            "<title>Coffee Shop</title>\n</head>\n<body>\n</body>\n</html>\n"
        const val IMAGE_PROMPT = "a cozy cabin in a snowy forest at night, warm lights, cinematic, highly detailed"
        const val VIDEO_PROMPT = "a golden retriever running through a sunflower field at sunset, slow motion"
        const val MUSIC_PRESET = "Upbeat 80s Synthwave synth bass & drums"
        const val VIBEVOICE_TRANSCRIPT = "What are three fun facts about octopuses?"
        const val TRANSLATOR_INPUT = "Good morning! Could you recommend a quiet café nearby where I can work for a few hours?"
    }
}
