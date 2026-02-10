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
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalUriHandler
import com.llmhub.llmhub.R
import com.llmhub.llmhub.repository.GithubRepository
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.platform.LocalContext
import com.llmhub.llmhub.data.ThemePreferences

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
    val uriHandler = LocalUriHandler.current
    val context = LocalContext.current
    val preferences = remember { ThemePreferences(context) }
    
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
                title = "feature_scam_detector",
                description = "feature_scam_detector_desc",
                icon = Icons.Filled.Security,
                gradient = Pair(Color(0xFFfa709a), Color(0xFFfee140)),
                route = "scam_detector"
            ),
            FeatureCard(
                title = "feature_image_generator",
                description = "feature_image_generator_desc",
                icon = Icons.Filled.Palette,
                gradient = Pair(Color(0xFF6a11cb), Color(0xFF2575fc)),
                route = "image_generator"
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
                        // App launcher icon - center cropped with equal edges removed
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .clip(RoundedCornerShape(8.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                painter = painterResource(id = R.mipmap.ic_launcher_foreground),
                                contentDescription = null,
                                modifier = Modifier
                                    .size(52.dp)
                                    .scale(1.6f),
                                tint = Color.Unspecified
                            )
                        }
                        Text(
                            text = stringResource(R.string.app_name),
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold
                        )
                    }
                },
                actions = {
                    // Chat History Button - Commented out
                    /*
                    IconButton(onClick = onNavigateToChatHistory) {
                        Icon(
                            imageVector = Icons.Outlined.History,
                            contentDescription = stringResource(R.string.drawer_recent_chats),
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                    */
                    
                    // GitHub Stars
                    val stars by preferences.githubStars.collectAsState(initial = 0)
                    
                    // Sync stars once
                    LaunchedEffect(Unit) {
                        GithubRepository.refreshStars(preferences)
                    }

                    if (stars > 0) {
                        Surface(
                            shape = CircleShape,
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                            modifier = Modifier
                                .padding(end = 8.dp)
                                .clip(CircleShape)
                                .clickable { uriHandler.openUri("https://github.com/timmyy123/LLM-Hub") }
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Icon(
                                    imageVector = rememberGithubIcon(),
                                    contentDescription = "GitHub Stars",
                                    modifier = Modifier.size(16.dp),
                                    tint = MaterialTheme.colorScheme.onSurface
                                )
                                Icon(
                                    imageVector = Icons.Filled.Star,
                                    contentDescription = null,
                                    modifier = Modifier.size(16.dp),
                                    tint = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = "$stars",
                                    style = MaterialTheme.typography.labelMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                            }
                        }
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
        BoxWithConstraints(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Use BoxWithConstraints sizes for robust orientation and rotation handling
            val isLandscapeLocal = maxWidth > maxHeight

            // Columns: landscape -> 3, portrait -> 2 (both phone & tablet portrait keep 2 columns)
            val columns = if (isLandscapeLocal) 3 else 2

            // Decide how many rows to fit on screen: landscape 2 rows (3x2), tablet portrait 3 rows (2x3)
            val headerEstimatedHeight = if (isLandscapeLocal) 120.dp else 160.dp
            val rowsWanted: Int? = when {
                isLandscapeLocal -> 2 // Show 3x2 in landscape
                !isLandscapeLocal && maxWidth >= 600.dp -> 3 // Tablet portrait: 2x3
                else -> null // Phone portrait: keep original scroll behavior
            }

            // Calculate card height so all rows fit on screen when rowsWanted is set
            val cardHeight: Dp? = rowsWanted?.let { rows ->
                val contentPaddingVertical = 32.dp // PaddingValues(16.dp) -> top+bottom
                val verticalSpacing = 16.dp * (rows - 1)
                val totalSpacing = contentPaddingVertical + headerEstimatedHeight + verticalSpacing
                val calc = (maxHeight - totalSpacing) / rows
                if (calc < 120.dp) 120.dp else calc
            }

            LazyVerticalGrid(
                columns = GridCells.Fixed(columns),
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
                        cardHeight = cardHeight,
                        onClick = {
                            onNavigateToFeature(feature.route)
                        }
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
                .padding(vertical = 12.dp, horizontal = 16.dp),
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
            Spacer(modifier = Modifier.height(8.dp))
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
    cardHeight: Dp? = null,
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
            modifier = run {
                var m: Modifier = Modifier.fillMaxWidth()
                m = if (cardHeight != null) m.height(cardHeight) else m.aspectRatio(1f)
                m = m.scale(scale)
                m = m.clickable(
                    interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                    indication = null
                ) {
                    onClick()
                }
                m
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
        "feature_scam_detector" -> R.string.feature_scam_detector
        "feature_scam_detector_desc" -> R.string.feature_scam_detector_desc
        "feature_image_generator" -> R.string.feature_image_generator
        "feature_image_generator_desc" -> R.string.feature_image_generator_desc
        else -> R.string.app_name
    }
}

@Composable
fun rememberGithubIcon(): ImageVector {
    return remember {
        ImageVector.Builder(
            name = "Github",
            defaultWidth = 24.dp,
            defaultHeight = 24.dp,
            viewportWidth = 24f,
            viewportHeight = 24f
        ).apply {
            path(
                fill = SolidColor(Color.Black),
                fillAlpha = 1f,
                stroke = null,
                strokeAlpha = 1f,
                pathFillType = PathFillType.NonZero
            ) {
                moveTo(12f, 0.297f)
                curveTo(5.37f, 0.297f, 0f, 5.67f, 0f, 12.297f)
                curveTo(0f, 17.6f, 3.438f, 22.097f, 8.205f, 23.682f)
                curveTo(8.805f, 23.795f, 9.025f, 23.424f, 9.025f, 23.105f)
                curveTo(9.025f, 22.82f, 9.015f, 22.065f, 9.01f, 21.065f)
                curveTo(5.672f, 21.789f, 4.968f, 19.462f, 4.968f, 19.462f)
                curveTo(4.422f, 18.076f, 3.633f, 17.7f, 3.633f, 17.7f)
                curveTo(2.546f, 16.959f, 3.717f, 16.974f, 3.717f, 16.974f)
                curveTo(4.922f, 17.059f, 5.556f, 18.213f, 5.556f, 18.213f)
                curveTo(6.641f, 20.07f, 8.402f, 19.533f, 9.092f, 19.222f)
                curveTo(9.2f, 18.436f, 9.514f, 17.9f, 9.86f, 17.597f)
                curveTo(7.198f, 17.294f, 4.4f, 16.265f, 4.4f, 11.67f)
                curveTo(4.4f, 10.36f, 4.868f, 9.293f, 5.637f, 8.453f)
                curveTo(5.513f, 8.15f, 5.101f, 6.925f, 5.755f, 5.275f)
                curveTo(5.755f, 5.275f, 6.763f, 4.953f, 9.057f, 6.505f)
                curveTo(10.015f, 6.239f, 11.042f, 6.106f, 12.065f, 6.111f)
                curveTo(13.088f, 6.106f, 14.115f, 6.239f, 15.074f, 6.505f)
                curveTo(17.366f, 4.953f, 18.373f, 5.275f, 18.373f, 5.275f)
                curveTo(19.028f, 6.925f, 18.617f, 8.15f, 18.494f, 8.453f)
                curveTo(19.265f, 9.293f, 19.73f, 10.36f, 19.73f, 11.67f)
                curveTo(19.73f, 16.275f, 16.927f, 17.29f, 14.258f, 17.587f)
                curveTo(14.695f, 17.964f, 15.085f, 18.708f, 15.085f, 19.847f)
                curveTo(15.085f, 21.47f, 15.07f, 22.784f, 15.07f, 23.105f)
                curveTo(15.07f, 23.427f, 15.293f, 23.804f, 15.898f, 23.681f)
                curveTo(20.662f, 22.093f, 24.1f, 17.597f, 24.1f, 12.297f)
                curveTo(24.1f, 5.67f, 18.73f, 0.297f, 12.1f, 0.297f)
                close()
            }
        }.build()
    }
}
