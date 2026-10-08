package com.llmhub.llmhub.screens

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
    SettingsItem(Icons.Default.Face, stringResource(R.string.tts_voice_setting),
        if (SupertonicModel.voiceFile(context, selectedVoice) != null) selectedVoice
        else stringResource(R.string.tts_please_download_voice)) { showVoices = true }
    SettingsItem(Icons.Default.Language, stringResource(R.string.supertonic_language),
        if (selectedLanguage.isEmpty()) stringResource(R.string.system_default_language)
        else Locale(selectedLanguage).getDisplayLanguage(Locale.getDefault())) { showLanguages = true }

    if (showVoices) AlertDialog(
        onDismissRequest = { showVoices = false },
        title = { Text(stringResource(R.string.tts_voice_setting)) },
        text = {
            LazyColumn {
                items(SupertonicModel.voices.keys.toList(), key = { it }) { key ->
                    val exists = remember(key, revision) { SupertonicModel.voiceFile(context, key) != null }
                    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(selectedVoice == key, onClick = {
                            if (exists && busy == null) scope.launch { preferences.setSupertonicVoice(key); showVoices = false }
                        }, enabled = exists && busy == null)
                        Column(Modifier.weight(1f)) {
                            Text(key, style = MaterialTheme.typography.bodyLarge)
                            SupertonicModel.voices[key]?.let { size ->
                                Text(android.text.format.Formatter.formatShortFileSize(context, size), style = MaterialTheme.typography.bodySmall)
                            }
                        }
                        if (busy == key) CircularProgressIndicator(Modifier.size(24.dp))
                        else if (exists) IconButton(enabled = busy == null, onClick = {
                            scope.launch {
                                withContext(Dispatchers.IO) { SupertonicModel.voiceFile(context, key)?.delete() }
                                if (key == selectedVoice) preferences.setSupertonicVoice("M1")
                                revision++
                            }
                        }) { Icon(Icons.Default.Delete, stringResource(R.string.delete)) }
                        else IconButton(enabled = busy == null, onClick = {
                            busy = key
                            scope.launch {
                                val client = HttpClient(Android)
                                try {
                                    ModelDownloader(client, context).downloadVoiceFile(ModelData.ttsModels.first { it.name == SupertonicModel.NAME }, key).collect { }
                                    preferences.setSupertonicVoice(key)
                                    revision++
                                } catch (e: CancellationException) { throw e
                                } catch (_: Exception) { error = context.getString(R.string.supertonic_voice_download_failed)
                                } finally { client.close(); busy = null }
                            }
                        }) { Icon(Icons.Default.Download, stringResource(R.string.download)) }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = { showVoices = false }) { Text(stringResource(R.string.cancel)) } }
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
