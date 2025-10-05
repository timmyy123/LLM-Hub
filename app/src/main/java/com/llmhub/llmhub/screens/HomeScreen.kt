package com.llmhub.llmhub.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.llmhub.llmhub.R
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

data class FeatureCard(
    val title: String,
    val description: String,
    val icon: ImageVector,
    val gradient: Pair<Color, Color>,
    val route: String
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onNavigateToFeature: (String) -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateToModels: () -> Unit,
    onNavigateToChatHistory: () -> Unit
) {
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.screenWidthDp > configuration.screenHeightDp
    
    // Feature cards data
    val features = remember {
        listOf(
            FeatureCard(
                title = "feature_ai_chat",
                description = "feature_ai_chat_desc",
                icon = Icons.Filled.Chat,
                gradient = Pair(Color(0xFF667eea), Color(0xFF764ba2)),
                route = "chat"
            ),
            FeatureCard(
                title = "feature_writing_aid",
                description = "feature_writing_aid_desc",
                icon = Icons.Filled.Edit,
                gradient = Pair(Color(0xFFf093fb), Color(0xFFf5576c)),
                route = "writing_aid"
            ),
            FeatureCard(
                title = "feature_translator",
                description = "feature_translator_desc",
                icon = Icons.Filled.Language,
                gradient = Pair(Color(0xFF4facfe), Color(0xFF00f2fe)),
                route = "translator"
            ),
            FeatureCard(
                title = "feature_transcriber",
                description = "feature_transcriber_desc",
                icon = Icons.Filled.Mic,
                gradient = Pair(Color(0xFF43e97b), Color(0xFF38f9d7)),
                route = "transcriber"
            ),
            FeatureCard(
                title = "feature_code_assistant",
                description = "feature_code_assistant_desc",
                icon = Icons.Filled.Code,
                gradient = Pair(Color(0xFFfa709a), Color(0xFFfee140)),
                route = "code_assistant"
            )
        )
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Psychology,
                            contentDescription = null,
                            modifier = Modifier.size(32.dp),
                            tint = MaterialTheme.colorScheme.primary
                        )
                        Text(
                            text = stringResource(R.string.app_name),
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold
                        )
                    }
                },
                actions = {
                    // Chat History Button
                    IconButton(onClick = onNavigateToChatHistory) {
                        Icon(
                            imageVector = Icons.Outlined.History,
                            contentDescription = stringResource(R.string.drawer_recent_chats),
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                    
                    // Models Button
                    IconButton(onClick = onNavigateToModels) {
                        Icon(
                            imageVector = Icons.Outlined.Download,
                            contentDescription = stringResource(R.string.download_models),
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                    
                    // Settings Button
                    IconButton(onClick = onNavigateToSettings) {
                        Icon(
                            imageVector = Icons.Outlined.Settings,
                            contentDescription = stringResource(R.string.settings),
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                contentPadding = PaddingValues(16.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.fillMaxSize()
            ) {
                // Header Section
                item(span = { androidx.compose.foundation.lazy.grid.GridItemSpan(maxLineSpan) }) {
                    AnimatedWelcomeHeader()
                }
                
                // Feature Cards
                itemsIndexed(features) { index, feature ->
                    AnimatedFeatureCard(
                        feature = feature,
                        index = index,
                        onClick = { onNavigateToFeature(feature.route) }
                    )
                }
            }
        }
    }
}

@Composable
fun AnimatedWelcomeHeader() {
    var visible by remember { mutableStateOf(false) }
    
    // Animate visibility on first composition
    LaunchedEffect(Unit) {
        visible = true
    }
    
    // Shimmer animation for the welcome text
    val infiniteTransition = rememberInfiniteTransition(label = "shimmer")
    val shimmerOffset by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1000f,
        animationSpec = infiniteRepeatable(
            animation = tween(3000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "shimmerOffset"
    )
    
    AnimatedVisibility(
        visible = visible,
        enter = fadeIn(
            animationSpec = tween(durationMillis = 600, easing = FastOutSlowInEasing)
        ) + slideInVertically(
            initialOffsetY = { -it / 2 },
            animationSpec = tween(durationMillis = 600, easing = FastOutSlowInEasing)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 24.dp, horizontal = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Animated gradient text for "Welcome to LLM Hub"
            Text(
                text = stringResource(R.string.home_welcome),
                style = MaterialTheme.typography.headlineMedium.copy(
                    brush = Brush.linearGradient(
                        colors = listOf(
                            Color(0xFF667eea),
                            Color(0xFF764ba2),
                            Color(0xFFf093fb),
                            Color(0xFF667eea)
                        ),
                        start = androidx.compose.ui.geometry.Offset(shimmerOffset, 0f),
                        end = androidx.compose.ui.geometry.Offset(shimmerOffset + 300f, 100f)
                    )
                ),
                fontWeight = FontWeight.ExtraBold,
                textAlign = TextAlign.Center,
                letterSpacing = 0.5.sp
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = stringResource(R.string.home_subtitle),
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun AnimatedFeatureCard(
    feature: FeatureCard,
    index: Int,
    onClick: () -> Unit
) {
    var visible by remember { mutableStateOf(false) }
    var isPressed by remember { mutableStateOf(false) }
    
    // Staggered entrance animation
    LaunchedEffect(Unit) {
        delay(index * 80L)
        visible = true
    }
    
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.95f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "scale"
    )
    
    AnimatedVisibility(
        visible = visible,
        enter = fadeIn(
            animationSpec = tween(durationMillis = 400, easing = FastOutSlowInEasing)
        ) + slideInVertically(
            initialOffsetY = { it / 3 },
            animationSpec = tween(durationMillis = 400, easing = FastOutSlowInEasing)
        )
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .scale(scale)
                .clickable(
                    interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                    indication = null
                ) {
                    onClick()
                },
            shape = RoundedCornerShape(24.dp),
            elevation = CardDefaults.cardElevation(
                defaultElevation = 4.dp,
                pressedElevation = 8.dp
            )
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        brush = Brush.linearGradient(
                            colors = listOf(
                                feature.gradient.first.copy(alpha = 0.9f),
                                feature.gradient.second.copy(alpha = 0.9f)
                            )
                        )
                    )
                    .padding(12.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    // Icon with circular background
                    Box(
                        modifier = Modifier
                            .size(56.dp)
                            .clip(CircleShape)
                            .background(Color.White.copy(alpha = 0.3f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = feature.icon,
                            contentDescription = null,
                            modifier = Modifier.size(32.dp),
                            tint = Color.White
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Title
                    Text(
                        text = stringResource(
                            id = getStringResourceId(feature.title)
                        ),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                        textAlign = TextAlign.Center,
                        maxLines = 2
                    )
                    
                    Spacer(modifier = Modifier.height(6.dp))
                    
                    // Description
                    Text(
                        text = stringResource(
                            id = getStringResourceId(feature.description)
                        ),
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.White.copy(alpha = 0.9f),
                        textAlign = TextAlign.Center,
                        maxLines = 3,
                        fontSize = 11.sp,
                        lineHeight = 14.sp
                    )
                }
            }
        }
    }
}

// Helper function to get string resource ID from string key
@Composable
private fun getStringResourceId(key: String): Int {
    return when (key) {
        "feature_ai_chat" -> R.string.feature_ai_chat
        "feature_ai_chat_desc" -> R.string.feature_ai_chat_desc
        "feature_writing_aid" -> R.string.feature_writing_aid
        "feature_writing_aid_desc" -> R.string.feature_writing_aid_desc
        "feature_translator" -> R.string.feature_translator
        "feature_translator_desc" -> R.string.feature_translator_desc
        "feature_transcriber" -> R.string.feature_transcriber
        "feature_transcriber_desc" -> R.string.feature_transcriber_desc
        "feature_code_assistant" -> R.string.feature_code_assistant
        "feature_code_assistant_desc" -> R.string.feature_code_assistant_desc
        else -> R.string.app_name
    }
}
