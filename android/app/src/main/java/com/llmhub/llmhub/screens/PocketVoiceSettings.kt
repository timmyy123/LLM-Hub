package com.llmhub.llmhub.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import com.llmhub.llmhub.data.PocketTtsModel
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.ui.components.PocketTtsEngine
import com.llmhub.llmhub.ui.components.PocketVoiceImport
import com.llmhub.llmhub.ui.components.PocketVoiceRecorder
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

/** Reference audio is copied privately and encoded locally; nothing is uploaded. */
@Composable
internal fun PocketVoiceSettings() {
    val context = LocalContext.current
    val preferences = remember { ThemePreferences(context) }
    val scope = rememberCoroutineScope()
    val selected by preferences.pocketVoice.collectAsState(initial = "")
    var open by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var revision by remember { mutableIntStateOf(0) }
    var error by remember { mutableStateOf<String?>(null) }
    var renamingVoice by remember { mutableStateOf<File?>(null) }
    var voiceName by remember { mutableStateOf("") }
    var savingName by remember { mutableStateOf(false) }
    var renameFailed by remember { mutableStateOf(false) }
    val voices = remember(revision) { PocketTtsModel.voices(context) }
    val selectedLabel = voices.find { it.name == selected }?.let(PocketTtsModel::label)
    var recording by remember { mutableStateOf(false) }
    var seconds by remember { mutableIntStateOf(0) }
    var captureJob by remember { mutableStateOf<Job?>(null) }
    val stopCapture = remember { AtomicBoolean(false) }
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP && recording) captureJob?.cancel()
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            captureJob?.cancel()
        }
    }
    fun cloneReference(createReference: suspend ((File) -> Unit) -> Pair<File, String>) {
        if (busy) return
        busy = true
        captureJob = scope.launch(start = kotlinx.coroutines.CoroutineStart.UNDISPATCHED) {
            var reference: File? = null
            var committed = false
            var wasRecording = false
            try {
                check(PocketTtsModel.isComplete(context))
                wasRecording = recording
                val (file, label) = createReference { reference = it }
                recording = false
                withContext(Dispatchers.IO) {
                    PocketTtsEngine(PocketTtsModel.directory(context), PocketTtsModel.voicesDirectory(context)).use { it.encode(file) }
                    coroutineContext.ensureActive()
                    File(file.path + ".txt").writeText(label.take(80))
                }
                coroutineContext.ensureActive()
                withContext(kotlinx.coroutines.NonCancellable) {
                    preferences.setPocketVoice(file.name)
                    committed = true
                }
                revision++
            } catch (e: CancellationException) { throw e
            } catch (e: Exception) {
                android.util.Log.e("PocketVoiceSettings", "Local voice creation failed", e)
                error = context.getString(if (wasRecording) R.string.pocket_record_failed else R.string.pocket_clone_failed)
            } finally {
                if (!committed) reference?.let { file ->
                    withContext(kotlinx.coroutines.NonCancellable + Dispatchers.IO) { PocketTtsModel.deleteVoice(context, file.name) }
                }
                recording = false
                busy = false
                captureJob = null
            }
        }
    }
    fun startRecording() {
        if (!open || busy || !lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) return
        seconds = 0
        stopCapture.set(false)
        recording = true
        cloneReference { track ->
            val file = File(PocketTtsModel.voicesDirectory(context), "${UUID.randomUUID()}.wav")
            track(file)
            PocketVoiceRecorder.record(file, stopCapture) { seconds = it }
            file to context.getString(R.string.pocket_recorded_voice,
                java.text.DateFormat.getDateTimeInstance(java.text.DateFormat.SHORT, java.text.DateFormat.SHORT).format(java.util.Date()))
        }
    }
    val microphonePermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) scope.launch {
            lifecycleOwner.lifecycle.currentStateFlow.first { it.isAtLeast(Lifecycle.State.RESUMED) }
            startRecording()
        } else error = context.getString(R.string.pocket_microphone_required)
    }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) cloneReference { track ->
            PocketVoiceImport.import(context, uri, PocketTtsModel.voicesDirectory(context), track)
        }
    }
    SettingsItem(Icons.Default.RecordVoiceOver, stringResource(R.string.pocket_voice_cloning),
        selectedLabel ?: stringResource(R.string.pocket_voice_help)) { open = true }
    if (open) AlertDialog(
        onDismissRequest = {
            if (recording) { captureJob?.cancel(); open = false }
            else if (!busy) open = false
        },
        title = { Text(stringResource(R.string.pocket_voice_cloning)) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(stringResource(R.string.pocket_voice_help))
                if (!PocketTtsModel.isComplete(context)) Text(stringResource(R.string.pocket_model_required), color = MaterialTheme.colorScheme.error)
                if (recording) {
                    Text(stringResource(R.string.pocket_recording, seconds))
                    FilledTonalButton(enabled = seconds >= 3, onClick = { stopCapture.set(true) }) {
                        Icon(Icons.Default.Stop, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.pocket_stop_and_clone))
                    }
                } else if (busy) Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(24.dp))
                    Spacer(Modifier.width(12.dp))
                    Text(stringResource(R.string.pocket_cloning))
                }
                if (!recording) OutlinedButton(enabled = !busy && PocketTtsModel.isComplete(context), onClick = {
                    if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) startRecording()
                    else microphonePermission.launch(Manifest.permission.RECORD_AUDIO)
                }) {
                    Icon(Icons.Default.Mic, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text(stringResource(R.string.pocket_record_voice))
                }
                LazyColumn(Modifier.heightIn(max = 240.dp)) {
                    items(voices, key = { it.name }) { voice ->
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            RadioButton(selected == voice.name, enabled = !busy && PocketTtsModel.isComplete(context), onClick = {
                                scope.launch {
                                    preferences.setPocketVoice(voice.name)
                                }
                            })
                            Text(PocketTtsModel.label(voice), Modifier.weight(1f))
                            IconButton(enabled = !busy, onClick = {
                                voiceName = PocketTtsModel.label(voice)
                                renameFailed = false
                                renamingVoice = voice
                            }) { Icon(Icons.Default.Edit, stringResource(R.string.action_rename)) }
                            IconButton(enabled = !busy, onClick = {
                                scope.launch {
                                    withContext(Dispatchers.IO) { PocketTtsModel.deleteVoice(context, voice.name) }
                                    if (selected == voice.name) preferences.setPocketVoice("")
                                    revision++
                                }
                            }) { Icon(Icons.Default.Delete, stringResource(R.string.delete)) }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(enabled = !busy && PocketTtsModel.isComplete(context), onClick = {
                picker.launch(arrayOf("audio/*", "application/octet-stream"))
            }) { Text(stringResource(R.string.pocket_import_voice)) }
        },
        dismissButton = { TextButton(enabled = !busy || recording, onClick = { captureJob?.cancel(); open = false }) { Text(stringResource(R.string.cancel)) } }
    )
    renamingVoice?.let { voice -> AlertDialog(
        onDismissRequest = { if (!savingName) renamingVoice = null },
        title = { Text(stringResource(R.string.action_rename)) },
        text = {
            OutlinedTextField(
                value = voiceName,
                onValueChange = { voiceName = it.take(80); renameFailed = false },
                label = { Text(stringResource(R.string.tts_voice_name)) },
                singleLine = true,
                enabled = !savingName,
                isError = renameFailed,
                supportingText = if (renameFailed) ({ Text(stringResource(R.string.tts_voice_rename_failed)) }) else null
            )
        },
        confirmButton = { TextButton(enabled = !savingName && voiceName.isNotBlank(), onClick = {
            savingName = true
            scope.launch {
                try {
                    withContext(Dispatchers.IO) { PocketTtsModel.renameVoice(context, voice.name, voiceName) }
                    revision++
                    renamingVoice = null
                } catch (e: CancellationException) { throw e
                } catch (e: Exception) {
                    android.util.Log.e("PocketVoiceSettings", "Voice rename failed", e)
                    renameFailed = true
                } finally { savingName = false }
            }
        }) { Text(stringResource(R.string.save)) } },
        dismissButton = { TextButton(enabled = !savingName, onClick = { renamingVoice = null }) {
            Text(stringResource(R.string.cancel))
        } }
    ) }
    error?.let { message -> AlertDialog(
        onDismissRequest = { error = null }, text = { Text(message) },
        confirmButton = { TextButton(onClick = { error = null }) { Text(stringResource(android.R.string.ok)) } }
    ) }
}
