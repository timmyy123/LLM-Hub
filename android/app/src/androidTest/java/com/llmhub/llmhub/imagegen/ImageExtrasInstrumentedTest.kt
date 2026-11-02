package com.llmhub.llmhub.imagegen

import android.net.Uri
import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.graphics.asAndroidBitmap
import com.llmhub.llmhub.screens.ImageGeneratorScreen
import com.llmhub.llmhub.screens.ImportExternalModelDialog
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.R
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import java.io.File

class ImageExtrasInstrumentedTest {
    @get:Rule val compose = createComposeRule()
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun translationsUseUnicodeWordBoundariesAndNeverCascade() {
        val options = ImageGenerationOptions(assets = listOf(ImageExtra(name = "translations", kind = "translations",
            mappings = mapOf("cat" to "dog", "dog" to "wolf", "猫" to "cat", "red cat" to "\$1\\cat"))))
        assertEquals("dog catfish cat_2 猫咪 cat, wolf; \$1\\cat", options.translatedPrompt("cat catfish cat_2 猫咪 猫, dog; red cat"))
        assertEquals("cat", options.copy(assets = options.assets.map { it.copy(enabled = false) }).translatedPrompt("cat"))
    }

    @Test fun importsPortablePromptResourcesAndPersistsSourceText() {
        val store = ImageExtrasStore(context)
        val original = store.load()
        val file = File(context.cacheDir, "modifier-test.txt")
        try {
            file.writeText("cat=dog\ndog=wolf")
            val translation = store.importFile(Uri.fromFile(file), "translations")
            file.writeText("cinematic\nsoft light")
            val tags = store.importFile(Uri.fromFile(file), "tags")
            file.writeText("{\"portrait\":\"a detailed portrait, soft lighting\"}")
            val definitions = store.importFile(Uri.fromFile(file), "definitions")
            val options = ImageGenerationOptions(negativePrompt = "cat", assets = listOf(translation, tags, definitions))
            store.save(options)
            assertEquals("cat", store.load().negativePrompt)
            assertEquals("dog", store.load().json().getString("negativePrompt"))
            assertTrue(store.load().promptItems().contains("portrait" to "a detailed portrait, soft lighting"))
            assertTrue(store.load().promptItems().contains("cinematic" to "cinematic"))
            file.writeText("missing separator")
            assertThrows(IllegalArgumentException::class.java) { store.importFile(Uri.fromFile(file), "translations") }
            file.writeText("x".repeat(2_000_001))
            assertThrows(IllegalArgumentException::class.java) { store.importFile(Uri.fromFile(file), "tags") }
        } finally { store.save(original); file.delete() }
    }

    @Test fun editorAdjustsGuidanceAndClipSkipForImportedCheckpoints() {
        val store = ImageExtrasStore(context)
        val original = store.load()
        try {
            store.save(ImageGenerationOptions())
            compose.setContent { MaterialTheme { ImageExtrasEditor(enabled = true, nativeCheckpointSelected = true, onChanged = {}) } }
            compose.onAllNodesWithText(context.getString(R.string.image_extras_title))[0].assertIsDisplayed()
            compose.onAllNodes(SemanticsMatcher.keyIsDefined(androidx.compose.ui.semantics.SemanticsProperties.ProgressBarRangeInfo))[0].performSemanticsAction(androidx.compose.ui.semantics.SemanticsActions.SetProgress) { it(9f) }
            compose.onAllNodes(SemanticsMatcher.keyIsDefined(androidx.compose.ui.semantics.SemanticsProperties.ProgressBarRangeInfo))[1].performSemanticsAction(androidx.compose.ui.semantics.SemanticsActions.SetProgress) { it(2f) }
            compose.runOnIdle { assertEquals(9f, store.load().guidance); assertEquals(2, store.load().clipSkip) }
            File(context.cacheDir, "image-modifiers-ui.png").outputStream().use {
                compose.onRoot().captureToImage().asAndroidBitmap().compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it)
            }
        } finally { store.save(original) }
    }

    @Test fun imageFilesImportThroughModelImportDialog() {
        compose.setContent { MaterialTheme { ImportExternalModelDialog(onDismiss = {}, onSearchHuggingFace = { _, _, _ -> emptyList() },
            onDownloadHuggingFace = { _, _, _, _, _, _, _ -> }, onImport = { _, _ -> true }) } }
        // The dialog body is a height-limited lazy list; fields below the fold are composed on scroll.
        val list = compose.onNode(hasScrollToIndexAction())
        list.performScrollToNode(hasText("gguf"))
        compose.onNodeWithText("gguf").performClick()
        compose.onNodeWithText("CLIP-G").assertExists()
        // The menu lists LLM formats first, so image types can sit below the popup's visible area.
        compose.onNodeWithText(context.getString(R.string.image_extras_checkpoint)).performScrollTo().performClick()
        compose.waitForIdle()
        File(context.cacheDir, "image-import-dialog.png").outputStream().use {
            compose.onNode(isDialog()).captureToImage().asAndroidBitmap().compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it)
        }
        compose.onAllNodesWithText("gguf").assertCountEquals(0)
        list.performScrollToNode(hasText(context.getString(R.string.image_extras_checkpoint)))
        compose.onNodeWithText(context.getString(R.string.image_extras_checkpoint)).assertIsDisplayed()
        // The list is clickable (to hide the keyboard), which merges plain Text children into it.
        compose.onNodeWithText(context.getString(R.string.image_extras_android_help), useUnmergedTree = true).assertIsDisplayed()
        compose.onAllNodesWithText(context.getString(R.string.search_huggingface)).assertCountEquals(0)
    }

    @Test fun legacyConvertedModelsDisableClipSkip() {
        compose.setContent { MaterialTheme { ImageExtrasEditor(enabled = true, nativeCheckpointSelected = false, onChanged = {}) } }
        compose.onAllNodes(SemanticsMatcher.keyIsDefined(androidx.compose.ui.semantics.SemanticsProperties.ProgressBarRangeInfo))[1].assertIsNotEnabled()
    }

    @Test fun modifiersAreOnlyInSettingsSheet() {
        compose.setContent { MaterialTheme { ImageGeneratorScreen(onNavigateBack = {}, onNavigateToModels = {}) } }
        compose.onAllNodesWithText(context.getString(R.string.image_extras_title)).assertCountEquals(0)
        compose.onNodeWithContentDescription("Settings").performClick()
        compose.onAllNodesWithText(context.getString(R.string.image_extras_title))[0].assertIsDisplayed()
    }

    @Test fun nativeImageEntryPointsRejectMissingCheckpoint() {
        val runtime = SdCppImageRuntime()
        try {
            assertThrows(IllegalStateException::class.java) { runtime.load(File(context.cacheDir, "missing.gguf").path, ImageGenerationOptions(), false) }
        } finally { runtime.close() }
    }
}
