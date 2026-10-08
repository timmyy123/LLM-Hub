package com.llmhub.llmhub.screens

import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.*
import io.ktor.client.HttpClient
import io.ktor.client.engine.android.Android
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

/** Preset voice downloads and language selection for Supertonic 3. */
@Composable
internal fun SupertonicVoiceSettings() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val preferences = remember { ThemePreferences(context) }
    val selectedVoice by preferences.supertonicVoice.collectAsState(initial = "M1")
    val selectedLanguage by preferences.supertonicLanguage.collectAsState(initial = "")
    var showVoices by remember { mutableStateOf(false) }
    var showLanguages by remember { mutableStateOf(false) }
    var revision by remember { mutableIntStateOf(0) }
    var busy by remember { mutableStateOf<String?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var importing by remember { mutableStateOf(false) }
    val customVoices = remember(revision) { SupertonicModel.customVoices(context) }
    val allVoices = SupertonicModel.voices.keys.map { it to it } + customVoices
    val downloadedVoices = remember(revision) {
        (SupertonicModel.voices.keys + customVoices.map { it.first }).filter { SupertonicModel.voiceFile(context, it) != null }.toSet()
    }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null && !importing) {
            importing = true
            scope.launch {
                var importedKey: String? = null
                var committed = false
                try {
                    withContext(Dispatchers.IO) {
                        val name = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
                            if (it.moveToFirst()) it.getString(0) else null
                        }?.take(80) ?: context.getString(R.string.tts_voice_setting)
                        context.contentResolver.openInputStream(uri)!!.use { input ->
                            importedKey = SupertonicModel.importVoice(context, input, name.removeSuffix(".json"))
                        }
                    }
                    withContext(kotlinx.coroutines.NonCancellable) {
                        preferences.setSupertonicVoice(checkNotNull(importedKey))
                        committed = true
                    }
                    revision++
                } catch (e: CancellationException) { throw e
                } catch (_: Exception) {
                    error = context.getString(R.string.supertonic_import_failed)
                } finally {
                    if (!committed) importedKey?.let { key ->
                        withContext(kotlinx.coroutines.NonCancellable + Dispatchers.IO) { SupertonicModel.deleteVoice(context, key) }
                    }
                    importing = false
                }
            }
        }
    }
    SettingsItem(Icons.Default.Face, stringResource(R.string.tts_voice_setting),
        if (SupertonicModel.voiceFile(context, selectedVoice) != null) SupertonicModel.voiceLabel(context, selectedVoice)
        else stringResource(R.string.tts_please_download_voice)) { showVoices = true }
    SettingsItem(Icons.Default.Language, stringResource(R.string.supertonic_language),
        if (selectedLanguage.isEmpty()) stringResource(R.string.system_default_language)
        else Locale(selectedLanguage).getDisplayLanguage(Locale.getDefault())) { showLanguages = true }

    if (showVoices) TtsVoiceDownloadDialog(
        voices = allVoices,
        selectedVoice = selectedVoice,
        downloadedVoices = downloadedVoices,
        downloadingVoice = busy,
        onSelect = { key ->
            scope.launch { preferences.setSupertonicVoice(key); showVoices = false }
        },
        onDelete = { key ->
            scope.launch {
                withContext(Dispatchers.IO) { SupertonicModel.deleteVoice(context, key) }
                if (key == selectedVoice) preferences.setSupertonicVoice("M1")
                revision++
            }
        },
        onDownload = { key ->
            busy = key
            scope.launch {
                val client = HttpClient(Android)
                try {
                    val token = com.llmhub.llmhub.viewmodels.ModelDownloadViewModel.getEffectiveToken(context)
                    ModelDownloader(client, context, token)
                        .downloadVoiceFile(ModelData.ttsModels.first { it.name == SupertonicModel.NAME }, key).collect { }
                    preferences.setSupertonicVoice(key)
                    revision++
                } catch (e: CancellationException) { throw e
                } catch (_: Exception) { error = context.getString(R.string.supertonic_voice_download_failed)
                } finally { client.close(); busy = null }
            }
        },
        onDismiss = { showVoices = false },
        importLabel = stringResource(R.string.supertonic_import_voice),
        importing = importing,
        onImport = { picker.launch(arrayOf("application/json", "text/json", "text/plain", "application/octet-stream")) }
    )
    if (showLanguages) AlertDialog(
        onDismissRequest = { showLanguages = false },
        title = { Text(stringResource(R.string.supertonic_language)) },
        text = {
            LazyColumn {
                items(listOf("") + SupertonicModel.languages) { lang ->
                    Row(Modifier.fillMaxWidth().clickable { scope.launch { preferences.setSupertonicLanguage(lang); showLanguages = false } },
                        verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(selectedLanguage == lang, onClick = { scope.launch { preferences.setSupertonicLanguage(lang); showLanguages = false } })
                        Text(if (lang.isEmpty()) stringResource(R.string.system_default_language) else Locale(lang).getDisplayLanguage(Locale.getDefault()))
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = { showLanguages = false }) { Text(stringResource(R.string.cancel)) } }
    )
    error?.let { message -> AlertDialog(
        onDismissRequest = { error = null }, text = { Text(message) },
        confirmButton = { TextButton(onClick = { error = null }) { Text(stringResource(android.R.string.ok)) } }
    ) }
}
