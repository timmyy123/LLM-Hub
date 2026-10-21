package com.llmhub.llmhub.imagegen

import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.text.font.FontWeight
import com.llmhub.llmhub.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun imageExtraLabel(kind: String): String = when (kind) {
    "checkpoint" -> stringResource(R.string.image_extras_checkpoint)
    "lora" -> "LoRA"
    "clipL" -> "CLIP-L"
    "clipG" -> "CLIP-G"
    "textEncoder" -> stringResource(R.string.image_extras_text_encoder)
    "vae" -> "VAE"
    "embedding" -> stringResource(R.string.image_extras_embedding)
    "tags" -> stringResource(R.string.image_extras_tags)
    "definitions" -> stringResource(R.string.image_extras_definitions)
    else -> stringResource(R.string.image_extras_translations)
}

@Composable
fun ImageExtrasEditor(enabled: Boolean, nativeCheckpointSelected: Boolean, onChanged: () -> Unit) {
    val context = LocalContext.current
    val store = remember { ImageExtrasStore(context) }
    var options by remember { mutableStateOf(store.load()) }
    fun save(value: ImageGenerationOptions) { options = value; store.save(value); onChanged() }
    val scope = rememberCoroutineScope()
    var addMenuOpen by remember { mutableStateOf(false) }
    var pendingKind by remember { mutableStateOf<String?>(null) }
    val invalidMessage = stringResource(R.string.image_extras_invalid)
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        val kind = pendingKind
        pendingKind = null
        if (uri == null || kind == null) return@rememberLauncherForActivityResult
        scope.launch {
            try {
                store.add(withContext(Dispatchers.IO) { store.importFile(uri, kind) })
                options = store.load()
                onChanged()
            } catch (e: Exception) {
                Toast.makeText(context, invalidMessage, Toast.LENGTH_LONG).show()
            }
        }
    }
    Card(
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp)
    ) {
      Column(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth().padding(16.dp)) {
        Text(stringResource(R.string.image_extras_title), style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
        OutlinedTextField(value = options.negativePrompt, onValueChange = { save(options.copy(negativePrompt = it)) },
            label = { Text(stringResource(R.string.image_extras_negative)) }, enabled = enabled, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp))
        Text("${stringResource(R.string.image_extras_guidance)}: ${"%.1f".format(options.guidance)}",
            style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
        Slider(value = options.guidance, onValueChange = { save(options.copy(guidance = it)) }, valueRange = 0f..20f, enabled = enabled)
        Text("${stringResource(R.string.image_extras_clip_skip)}: ${options.clipSkip}",
            style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
        Slider(value = options.clipSkip.toFloat(), onValueChange = { save(options.copy(clipSkip = it.toInt())) },
            valueRange = 1f..12f, steps = 10, enabled = enabled && nativeCheckpointSelected)
        // Add-ons are imported here, next to the model they modify. LoRA, encoders, VAE and
        // textual inversion need an imported checkpoint selected; prompt lists work with any model.
        Box {
            OutlinedButton(onClick = { addMenuOpen = true }, enabled = enabled, modifier = Modifier.fillMaxWidth()) {
                Icon(Icons.Default.Add, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text(stringResource(R.string.image_extras_add))
            }
            DropdownMenu(expanded = addMenuOpen, onDismissRequest = { addMenuOpen = false }) {
                ImageExtrasStore.addOnKinds.forEach { kind ->
                    DropdownMenuItem(
                        text = { Text(imageExtraLabel(kind)) },
                        enabled = nativeCheckpointSelected || kind !in ImageExtrasStore.checkpointOnlyKinds,
                        onClick = {
                            addMenuOpen = false
                            pendingKind = kind
                            picker.launch(arrayOf("*/*"))
                        }
                    )
                }
            }
        }
        if (!nativeCheckpointSelected) {
            Text(stringResource(R.string.image_extras_android_help), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        options.assets.forEach { asset ->
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(asset.name)
                        Text(imageExtraLabel(asset.kind), style = MaterialTheme.typography.bodySmall)
                    }
                    if (asset.kind != "checkpoint") Switch(checked = asset.enabled, onCheckedChange = { value ->
                        save(options.copy(assets = options.assets.map {
                            when {
                                it.id == asset.id -> it.copy(enabled = value)
                                value && asset.kind in listOf("clipL", "clipG", "textEncoder", "vae") && it.kind == asset.kind -> it.copy(enabled = false)
                                else -> it
                            }
                        }))
                    }, enabled = enabled && (nativeCheckpointSelected || asset.kind in listOf("tags", "definitions", "translations")))
                    IconButton(onClick = { save(options.copy(assets = options.assets.filterNot { it.id == asset.id })); store.remove(asset) }, enabled = enabled) {
                        Icon(Icons.Default.Delete, contentDescription = stringResource(R.string.delete))
                    }
                }
                if (asset.kind == "lora") {
                    Text("${stringResource(R.string.image_extras_weight)}: ${"%.2f".format(asset.weight)}",
                        style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                    Slider(value = asset.weight, onValueChange = { weight -> save(options.copy(assets = options.assets.map { if (it.id == asset.id) it.copy(weight = weight) else it })) },
                        valueRange = -2f..2f, enabled = enabled && nativeCheckpointSelected)
                }
            }
        }
      }
    }
}

@Composable
fun ImagePromptResources(refresh: Int, prompt: String, onPromptChanged: (String) -> Unit) {
    val context = LocalContext.current
    val items = remember(refresh) { ImageExtrasStore(context).load().promptItems() }
    if (items.isNotEmpty()) LazyRow(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(items) { (name, value) -> AssistChip(onClick = { onPromptChanged(prompt + (if (prompt.isBlank()) "" else ", ") + value) }, label = { Text(name) }) }
    }
}
