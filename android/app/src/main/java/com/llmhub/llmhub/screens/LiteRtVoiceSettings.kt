package com.llmhub.llmhub.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LiteRtTtsModel
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.viewmodels.ThemeViewModel
import kotlinx.coroutines.launch

/** Voice picker for the LiteRT-LM speech bundles. Every voice ships inside the model: nothing to download. */
@Composable
internal fun LiteRtVoiceSettings(model: String, themeViewModel: ThemeViewModel) {
    val context = LocalContext.current
    val preferences = remember { ThemePreferences(context) }
    val scope = rememberCoroutineScope()
    val isPocket = LiteRtTtsModel.isPocket(model)
    val kokoroVoice by themeViewModel.selectedTtsVoice.collectAsState()
    val pocketVoice by preferences.pocketLiteRtVoice.collectAsState(initial = LiteRtTtsModel.POCKET_DEFAULT_VOICE)
    val voices = remember(model) { LiteRtTtsModel.voices(model) }
    val selected = LiteRtTtsModel.resolveVoice(model, if (isPocket) pocketVoice else kokoroVoice)
    var open by remember { mutableStateOf(false) }

    SettingsItem(
        icon = Icons.Default.Face,
        title = stringResource(R.string.tts_voice_setting),
        subtitle = voices.find { it.first == selected }?.second ?: selected,
        onClick = { open = true }
    )

    if (open) {
        fun select(voice: String) {
            if (isPocket) scope.launch { preferences.setPocketLiteRtVoice(voice) }
            else themeViewModel.setSelectedTtsVoice(voice)
            open = false
        }
        AlertDialog(
            onDismissRequest = { open = false },
            title = { Text(stringResource(R.string.tts_voice_setting)) },
            text = {
                LazyColumn {
                    items(voices, key = { it.first }) { (key, label) ->
                        Row(
                            modifier = Modifier.fillMaxWidth().height(56.dp).clickable { select(key) },
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(selected = selected == key, onClick = { select(key) })
                            Spacer(Modifier.width(8.dp))
                            Text(label, style = MaterialTheme.typography.bodyLarge)
                        }
                    }
                }
            },
            confirmButton = { TextButton(onClick = { open = false }) { Text(stringResource(R.string.cancel)) } }
        )
    }
}
