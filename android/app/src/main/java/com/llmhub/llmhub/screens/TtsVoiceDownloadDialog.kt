package com.llmhub.llmhub.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.llmhub.llmhub.R

/** The existing Kokoro panel layout, shared by every downloadable preset voice engine. */
@Composable
internal fun TtsVoiceDownloadDialog(
    voices: List<Pair<String, String>>,
    selectedVoice: String,
    downloadedVoices: Set<String>,
    downloadingVoice: String?,
    onSelect: (String) -> Unit,
    onDelete: (String) -> Unit,
    onDownload: (String) -> Unit,
    onDismiss: () -> Unit,
    importLabel: String? = null,
    importing: Boolean = false,
    onImport: (() -> Unit)? = null
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.tts_voice_setting)) },
        text = {
            LazyColumn {
                if (downloadedVoices.isEmpty()) item {
                    Text(
                        text = stringResource(R.string.tts_please_download_voice),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                }
                items(voices, key = { it.first }) { (key, label) ->
                    val downloaded = key in downloadedVoices
                    val downloading = downloadingVoice == key
                    Row(
                        modifier = Modifier.fillMaxWidth().height(56.dp)
                            .clickable(enabled = downloaded && !downloading && !importing) { onSelect(key) },
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(selected = selectedVoice == key, onClick = {
                            if (downloaded && !downloading && !importing) onSelect(key)
                        })
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = label,
                            style = MaterialTheme.typography.bodyLarge,
                            modifier = Modifier.weight(1f),
                            color = if (downloaded) MaterialTheme.colorScheme.onSurface
                                else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(Modifier.width(8.dp))
                        val buttonModifier = Modifier.height(40.dp).widthIn(min = 120.dp)
                        val buttonPadding = PaddingValues(horizontal = 12.dp, vertical = 0.dp)
                        if (downloaded) {
                            OutlinedButton(
                                onClick = { onDelete(key) },
                                enabled = !importing,
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.error),
                                modifier = buttonModifier,
                                contentPadding = buttonPadding
                            ) {
                                Icon(Icons.Default.Delete, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(Modifier.width(4.dp))
                                Text(stringResource(R.string.delete), style = MaterialTheme.typography.labelLarge)
                            }
                        } else {
                            Button(
                                onClick = { if (downloadingVoice == null) onDownload(key) },
                                enabled = downloadingVoice == null && !importing,
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                                modifier = buttonModifier,
                                contentPadding = buttonPadding
                            ) {
                                if (downloading) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
                                else {
                                    Icon(Icons.Default.CloudDownload, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(Modifier.width(4.dp))
                                    Text(stringResource(R.string.download), style = MaterialTheme.typography.labelLarge)
                                }
                            }
                        }
                    }
                }
            }
        },
        dismissButton = {
            if (onImport != null && importLabel != null) TextButton(enabled = !importing && downloadingVoice == null, onClick = onImport) {
                if (importing) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
                else Text(importLabel)
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.cancel)) } }
    )
}
