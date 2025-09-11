package com.llmhub.llmhub.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.llmhub.llmhub.R

/**
 * Composable that shows RAG (document chat) status and context information.
 * Displays when documents are available and provides visual feedback for semantic search.
 */
@Composable
fun RagStatusIndicator(
    hasDocuments: Boolean,
    documentCount: Int = 0,
    isSearching: Boolean = false,
    foundContextChunks: Int = 0,
    modifier: Modifier = Modifier
) {
    if (hasDocuments || isSearching || foundContextChunks > 0) {
        Card(
            modifier = modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.6f)
            ),
            shape = RoundedCornerShape(12.dp)
        ) {
            Row(
                modifier = Modifier
                    .padding(12.dp)
                    .fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Document icon
                Icon(
                    imageVector = if (isSearching) Icons.Default.Search else Icons.Default.Description,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(16.dp)
                )
                
                // Status text
                Column(modifier = Modifier.weight(1f)) {
                    if (isSearching) {
                        Text(
                            text = stringResource(R.string.searching_documents),
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.Medium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    } else if (foundContextChunks > 0) {
                        Text(
                            text = stringResource(R.string.using_document_context_format, foundContextChunks),
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.Medium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    } else if (hasDocuments) {
                        Text(
                            text = stringResource(R.string.documents_available_format, documentCount),
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.Medium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                        Text(
                            text = stringResource(R.string.ai_can_reference_documents),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f),
                            fontSize = 11.sp
                        )
                    }
                }
                
                // Loading indicator when searching
                if (isSearching) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(14.dp),
                        strokeWidth = 1.5.dp,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }
    }
}

/**
 * Compact version for use in message input area
 */
@Composable
fun CompactRagIndicator(
    hasDocuments: Boolean,
    documentCount: Int = 0,
    modifier: Modifier = Modifier
) {
    if (hasDocuments) {
        Row(
            modifier = modifier
                .clip(RoundedCornerShape(8.dp))
                .background(MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f))
                .padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Description,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(12.dp)
            )
            // Numeric document count removed to avoid showing replicated/global memory
            // chunks as an "attachment" badge. Keep the compact icon only.
        }
    }
}
