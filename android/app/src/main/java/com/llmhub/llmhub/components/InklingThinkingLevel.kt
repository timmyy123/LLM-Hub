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
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import com.llmhub.llmhub.R
import com.llmhub.llmhub.inference.InklingFormat
import kotlin.math.abs

private val LEVEL_LABELS = listOf(
    R.string.thinking_level_off,
    R.string.thinking_level_minimal,
    R.string.thinking_level_low,
    R.string.thinking_level_medium,
    R.string.thinking_level_high,
    R.string.thinking_level_max,
)

/**
 * Inkling 2's thinking level in place of the thinking switch: Off turns thinking off, the other levels turn it on
 * with that effort (InklingFormat.EFFORT_LEVELS).
 */
@Composable
internal fun InklingThinkingLevelRow(
    thinkingEnabled: Boolean,
    effort: Float,
    onSelect: (thinkingEnabled: Boolean, effort: Float) -> Unit,
) {
    val levels = InklingFormat.EFFORT_LEVELS
    val current = if (!thinkingEnabled) 0 else levels.indices.drop(1).minBy { abs(levels[it] - effort) }
    var expanded by remember { mutableStateOf(false) }
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(text = stringResource(R.string.thinking_level), style = MaterialTheme.typography.bodyMedium)
        Box {
            TextButton(onClick = { expanded = true }) {
                Text(stringResource(LEVEL_LABELS[current]))
                Icon(Icons.Default.ArrowDropDown, contentDescription = null)
            }
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                levels.forEachIndexed { index, level ->
                    DropdownMenuItem(
                        text = { Text(stringResource(LEVEL_LABELS[index])) },
                        onClick = {
                            expanded = false
                            onSelect(level > 0f, if (level > 0f) level else effort)
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
