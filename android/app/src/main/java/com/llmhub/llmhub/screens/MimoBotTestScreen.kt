package com.llmhub.llmhub.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
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
import com.llmhub.llmhub.mimobot.speech.KokoroTts
import com.llmhub.llmhub.mimobot.speech.SystemTts
import com.llmhub.llmhub.mimobot.speech.Tts
import com.llmhub.llmhub.mimobot.speech.kokoro.G2P
import com.llmhub.llmhub.mimobot.speech.kokoro.KokoroAssets
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

private enum class TtsChoice { System, Kokoro }
private const val DEFAULT_KOKORO_VOICE = "af_heart"

/**
 * Dev-mode voice-loop screen. Uses the phone's own mic + speaker to exercise
 * the full STT → LLM → TTS path before BLE transport / Whisper land.
 *
 * TTS is selectable: the system recognizer (works immediately) or Kokoro
 * (requires a one-time ~165 MB model download, runs through ONNX Runtime
 * with NNAPI EP).
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

    var ttsChoice by remember { mutableStateOf(TtsChoice.System) }

    // Track Kokoro asset readiness reactively.
    var kokoroReady by remember {
        mutableStateOf(KokoroAssets.isReady(context, DEFAULT_KOKORO_VOICE))
    }
    var downloadStage by remember { mutableStateOf<String?>(null) }
    var downloadDone by remember { mutableStateOf(0L) }
    var downloadTotal by remember { mutableStateOf(-1L) }

    // Build pipeline whenever the user model or TTS choice changes.
    val pipeline = remember(loadedModel, ttsChoice, kokoroReady) {
        if (loadedModel == null) return@remember null
        val tts: Tts = when (ttsChoice) {
            TtsChoice.System -> SystemTts(context)
            TtsChoice.Kokoro -> if (kokoroReady) {
                KokoroTts(
                    context = context,
                    modelPath = KokoroAssets.modelFile(context).absolutePath,
                    voicePackPath = KokoroAssets.voiceFile(context, DEFAULT_KOKORO_VOICE).absolutePath,
                    voiceId = DEFAULT_KOKORO_VOICE,
                )
            } else {
                // Kokoro picked but assets missing — fall back so the screen still works.
                SystemTts(context)
            }
        }
        VoicePipeline(
            scope = scope,
            inference = inference,
            model = loadedModel,
            stt = AndroidSpeechRecognizerStt(context),
            tts = tts,
            sink = SpeakerSink(),
        )
    }

    DisposableEffect(pipeline) { onDispose { pipeline?.cancel() } }

    val state by (pipeline?.state?.collectAsState() ?: remember { mutableStateOf(VoicePipeline.State.IDLE) })
    val transcript by (pipeline?.lastTranscript?.collectAsState() ?: remember { mutableStateOf("") })
    val response by (pipeline?.lastResponse?.collectAsState() ?: remember { mutableStateOf("") })
    val history by (pipeline?.history?.collectAsState() ?: remember { mutableStateOf(emptyList<VoicePipeline.Turn>()) })

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
                    Text("Voice", style = MaterialTheme.typography.labelMedium)
                    TtsChoice.values().forEach { choice ->
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .selectable(
                                    selected = ttsChoice == choice,
                                    onClick = { ttsChoice = choice },
                                )
                                .padding(vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            RadioButton(selected = ttsChoice == choice, onClick = { ttsChoice = choice })
                            val label = when (choice) {
                                TtsChoice.System -> "System (Android TTS)"
                                TtsChoice.Kokoro -> "Kokoro-82M (neural)"
                            }
                            Text(label)
                        }
                    }

                    if (ttsChoice == TtsChoice.Kokoro && !kokoroReady) {
                        Spacer(Modifier.height(8.dp))
                        if (downloadStage == null) {
                            Button(
                                modifier = Modifier.fillMaxWidth(),
                                onClick = {
                                    scope.launch {
                                        try {
                                            KokoroAssets.ensure(context, DEFAULT_KOKORO_VOICE)
                                                .collectLatest { p ->
                                                    downloadStage = p.stage
                                                    downloadDone = p.bytesDone
                                                    downloadTotal = p.bytesTotal
                                                }
                                            kokoroReady = KokoroAssets.isReady(context, DEFAULT_KOKORO_VOICE)
                                        } catch (t: Throwable) {
                                            downloadStage = "Download failed: ${t.message}"
                                        }
                                    }
                                },
                            ) { Text("Download Kokoro (~165 MB)") }
                        } else {
                            Text(downloadStage ?: "", style = MaterialTheme.typography.bodyMedium)
                            if (downloadTotal > 0) {
                                LinearProgressIndicator(
                                    progress = { (downloadDone.toFloat() / downloadTotal.toFloat()).coerceIn(0f, 1f) },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            } else {
                                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                            }
                        }
                    } else if (ttsChoice == TtsChoice.Kokoro) {
                        Spacer(Modifier.height(4.dp))
                        val g2pName = remember { G2P.best(context).displayName }
                        Text(
                            "G2P: $g2pName",
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                        if (g2pName.startsWith("dictionary")) {
                            Text(
                                "Bundled dictionary covers ~150 words. Run scripts/build_espeak_android.sh and rebuild for full coverage — see docs/espeak-ng-setup.md.",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }
                }
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Row(
                        Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("Conversation memory", style = MaterialTheme.typography.labelMedium)
                            Text(
                                if (history.isEmpty()) "no turns yet" else "${history.size} turn${if (history.size == 1) "" else "s"}",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                        if (history.isNotEmpty()) {
                            OutlinedButton(onClick = { pipeline?.clearHistory() }) { Text("Clear") }
                        }
                    }
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
