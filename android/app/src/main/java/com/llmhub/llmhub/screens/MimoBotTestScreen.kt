package com.llmhub.llmhub.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.llmhub.llmhub.LlmHubApplication
import com.llmhub.llmhub.R
import com.llmhub.llmhub.mimobot.audio.SpeakerSink
import com.llmhub.llmhub.mimobot.pipeline.VoicePipeline
import com.llmhub.llmhub.mimobot.speech.AndroidSpeechRecognizerStt
import com.llmhub.llmhub.mimobot.speech.SystemTts

/**
 * Dev-mode voice-loop screen. Uses the phone's own mic + speaker to exercise
 * the full STT → LLM → TTS path before the BLE transport and Whisper/Kokoro
 * backends are in place.
 *
 * Requires a model to be already loaded via the regular chat flow.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MimoBotTestScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val app = context.applicationContext as LlmHubApplication
    val scope = rememberCoroutineScope()

    val inference = remember { app.inferenceService }
    val loadedModel = remember { inference.getCurrentlyLoadedModel() }

    // Build pipeline once. DisposableEffect tears it down when we leave the screen.
    val pipeline = remember(loadedModel) {
        loadedModel?.let { model ->
            VoicePipeline(
                scope = scope,
                inference = inference,
                model = model,
                stt = AndroidSpeechRecognizerStt(context),
                tts = SystemTts(context),
                sink = SpeakerSink(),
            )
        }
    }

    DisposableEffect(pipeline) {
        onDispose {
            pipeline?.cancel()
        }
    }

    val state by (pipeline?.state?.collectAsState() ?: remember { mutableStateOf(VoicePipeline.State.IDLE) })
    val transcript by (pipeline?.lastTranscript?.collectAsState() ?: remember { mutableStateOf("") })
    val response by (pipeline?.lastResponse?.collectAsState() ?: remember { mutableStateOf("") })

    // Mic permission handling
    var hasMicPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasMicPermission = granted }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Mimo Bot (test)") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
            )
        },
    ) { inner ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(inner)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            if (loadedModel == null) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("No model loaded", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        Spacer(Modifier.height(8.dp))
                        Text("Open a chat first to load a model, then come back here.", style = MaterialTheme.typography.bodyMedium)
                    }
                }
                return@Column
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("Model", style = MaterialTheme.typography.labelMedium)
                    Text(loadedModel.name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                }
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("State", style = MaterialTheme.typography.labelMedium)
                    Text(state.name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                }
            }

            if (transcript.isNotEmpty()) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("You said", style = MaterialTheme.typography.labelMedium)
                        Text(transcript, style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }

            if (response.isNotEmpty()) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Reply", style = MaterialTheme.typography.labelMedium)
                        Text(response, style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }

            Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                if (state == VoicePipeline.State.IDLE) {
                    Button(
                        enabled = hasMicPermission,
                        onClick = { pipeline?.startTurn() },
                    ) { Text("Talk") }
                } else {
                    OutlinedButton(onClick = { pipeline?.cancel() }) { Text("Stop") }
                }
            }

            if (!hasMicPermission) {
                Button(
                    modifier = Modifier.fillMaxWidth(),
                    onClick = { permissionLauncher.launch(Manifest.permission.RECORD_AUDIO) },
                ) { Text("Grant microphone permission") }
            }
        }
    }
}
