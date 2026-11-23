package com.llmhub.llmhub.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelPreferences
import com.llmhub.llmhub.data.ThinkingLevels
import com.llmhub.llmhub.data.defaultConfigForModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlin.math.abs

/**
 * A thinking level in place of the thinking switch, for models whose template takes one (ThinkingLevels). The first
 * choice (None; for Inkling 2 it asks for effort 0 and may still think) sends the model's thinking-off
 * prompt; the other levels turn thinking on at that level.
 */
@Composable
internal fun ThinkingLevelRow(
    spec: ThinkingLevels,
    thinkingEnabled: Boolean,
    effort: Float,
    onSelect: (thinkingEnabled: Boolean, effort: Float) -> Unit,
) {
    val levels = spec.levels
    val resolved = spec.resolve(effort)
    val current = if (!thinkingEnabled) 0 else levels.indices.drop(1).minBy { abs(levels[it].effort - resolved) }
    var expanded by remember { mutableStateOf(false) }
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(text = stringResource(R.string.thinking_level), style = MaterialTheme.typography.bodyMedium)
        Box {
            TextButton(onClick = { expanded = true }) {
                Text(stringResource(levels[current].labelRes))
                Icon(Icons.Default.ArrowDropDown, contentDescription = null)
            }
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                levels.forEachIndexed { index, level ->
                    DropdownMenuItem(
                        text = { Text(stringResource(level.labelRes)) },
                        onClick = {
                            expanded = false
                            onSelect(index > 0, if (index > 0) level.effort else resolved)
                        },
                        trailingIcon = {
                            if (index == current) {
                                Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                            }
                        },
                    )
                }
            }
        }
    }
}

/**
 * A feature screen's thinking control: the level picker for models with thinking levels (the level is saved in the
 * model's config, which the next prompt reads, as the settings sheets do), else the thinking switch.
 */
@Composable
internal fun ThinkingSwitchOrLevel(
    model: LLMModel?,
    enableThinking: Boolean,
    onEnableThinking: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    val spec = ThinkingLevels.forModel(model)
    if (model == null || spec == null) {
        Row(
            modifier = modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = stringResource(R.string.enable_thinking),
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
            )
            Switch(checked = enableThinking, onCheckedChange = onEnableThinking)
        }
        return
    }
    val prefs = ModelPreferences(LocalContext.current)
    val scope = rememberCoroutineScope()
    var effort by remember(model.name) { mutableFloatStateOf(spec.default) }
    LaunchedEffect(model.name) { effort = spec.resolve(prefs.getModelConfig(model.name)?.thinkingEffort) }
    Box(modifier = modifier) {
        ThinkingLevelRow(spec, enableThinking, effort) { enabled, level ->
            onEnableThinking(enabled)
            effort = level
            scope.launch(Dispatchers.IO) {
                try {
                    val cfg = (prefs.getModelConfig(model.name) ?: defaultConfigForModel(model)).copy(thinkingEffort = level)
                    prefs.setModelConfig(model.name, cfg)
                } catch (_: Exception) {}
            }
        }
    }
}
