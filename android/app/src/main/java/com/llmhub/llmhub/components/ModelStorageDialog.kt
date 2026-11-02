package com.llmhub.llmhub.components

import android.text.format.Formatter
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.ModelStorage
import kotlinx.coroutines.launch

/** Chooses internal storage or the SD card for downloaded models and moves the ones already there. */
@Composable
fun ModelStorageDialog(onDismiss: () -> Unit, onMoved: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val onSdCard = remember { ModelStorage.isOnSdCard(context) }
    var useSdCard by remember { mutableStateOf(onSdCard) }
    var progress by remember { mutableStateOf<Float?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    val used = remember { ModelStorage.usedBytes(context) }
    fun size(bytes: Long) = Formatter.formatShortFileSize(context, bytes)

    AlertDialog(
        onDismissRequest = { if (progress == null) onDismiss() },
        title = { Text(stringResource(R.string.model_storage_title)) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Column(Modifier.selectableGroup()) {
                    listOf(false, true).forEach { sdCard ->
                        Row(
                            Modifier.fillMaxWidth().selectable(
                                selected = useSdCard == sdCard, enabled = progress == null, role = Role.RadioButton,
                                onClick = { useSdCard = sdCard }
                            ),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(selected = useSdCard == sdCard, onClick = null, enabled = progress == null)
                            Column {
                                Text(stringResource(if (sdCard) R.string.model_storage_sd_card else R.string.model_storage_internal))
                                Text(
                                    stringResource(R.string.model_storage_free, size(ModelStorage.freeBytes(context, sdCard))),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }
                Text(stringResource(R.string.model_storage_note, size(used)), style = MaterialTheme.typography.bodySmall)
                progress?.let {
                    Text(stringResource(R.string.model_storage_moving, (it * 100).toInt()))
                    LinearProgressIndicator(progress = { it }, modifier = Modifier.fillMaxWidth())
                }
                error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            }
        },
        confirmButton = {
            TextButton(
                enabled = progress == null && useSdCard != onSdCard,
                onClick = {
                    error = null
                    progress = 0f
                    scope.launch {
                        ModelStorage.move(context, useSdCard) { progress = it }
                            .onSuccess { progress = null; onMoved(); onDismiss() }
                            .onFailure { progress = null; error = context.getString(R.string.model_storage_failed, it.message.orEmpty()) }
                    }
                }
            ) { Text(stringResource(R.string.model_storage_move)) }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, enabled = progress == null) { Text(stringResource(R.string.cancel)) }
        }
    )
}
