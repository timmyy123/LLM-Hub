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
            ),
            FeatureCard(
                title = "feature_vibe_coder",
                description = "feature_vibe_coder_desc",
                icon = Icons.Filled.Code,
                gradient = Pair(Color(0xFFf794a4), Color(0xFFfdd6bd)),
                route = "vibe_coder"
            ),
            FeatureCard(
                title = "feature_creator_generation",
                description = "feature_creator_generation_desc",
                icon = Icons.Filled.AutoAwesome,
                gradient = Pair(Color(0xFF8EC5FC), Color(0xFFE0C3FC)),
                route = "creator_generation"
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
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    // Description
                    Text(
                        text = stringResource(
                            id = getStringResourceId(feature.description)
                        ),
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.White.copy(alpha = 0.8f),
                        textAlign = TextAlign.Center,
                        lineHeight = 16.sp,
                        modifier = Modifier.padding(horizontal = 8.dp)
                    )
                }
            }
        }
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
                strokeLineWidth = 1f,
                strokeLineCap = androidx.compose.ui.graphics.StrokeCap.Butt,
                strokeLineJoin = androidx.compose.ui.graphics.StrokeJoin.Miter,
                strokeLineMiter = 1f,
                pathFillType = PathFillType.NonZero
            ) {
                moveTo(12.0f, 2.0f)
                arcTo(10.0f, 10.0f, 0.0f, false, false, 2.0f, 12.0f)
                curveToRelative(0.0f, 4.42f, 2.87f, 8.17f, 6.84f, 9.5f)
                curveToRelative(0.5f, 0.09f, 0.68f, -0.22f, 0.68f, -0.48f)
                verticalLineToRelative(-1.7f)
                curveToRelative(-2.78f, 0.6f, -3.37f, -1.34f, -3.37f, -1.34f)
                curveToRelative(-0.45f, -1.16f, -1.11f, -1.47f, -1.11f, -1.47f)
                curveToRelative(-0.9f, -0.62f, 0.07f, -0.6f, 0.07f, -0.6f)
                curveToRelative(1.0f, 0.07f, 1.53f, 1.03f, 1.53f, 1.03f)
                curveToRelative(0.89f, 1.52f, 2.34f, 1.08f, 2.91f, 0.83f)
                curveToRelative(0.09f, -0.65f, 0.35f, -1.08f, 0.63f, -1.34f)
                curveToRelative(-2.22f, -0.25f, -4.55f, -1.11f, -4.55f, -4.94f)
                curveToRelative(0.0f, -1.1f, 0.39f, -1.99f, 1.03f, -2.69f)
                curveToRelative(-0.1f, -0.25f, -0.45f, -1.27f, 0.1f, -2.65f)
                curveToRelative(0.0f, 0.0f, 0.84f, -0.27f, 2.75f, 1.02f)
                curveTo(11.0f, 5.0f, 12.0f, 5.0f, 13.0f, 5.0f)
                curveToRelative(1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f)
                curveToRelative(1.91f, -1.29f, 2.75f, -1.02f, 2.75f, -1.02f)
                curveToRelative(0.55f, 1.38f, 0.2f, 2.4f, 0.1f, 2.65f)
                curveToRelative(0.64f, 0.7f, 1.03f, 1.6f, 1.03f, 2.69f)
                curveToRelative(0.0f, 3.84f, -2.34f, 4.68f, -4.57f, 4.93f)
                curveToRelative(0.36f, 0.31f, 0.68f, 0.92f, 0.68f, 1.85f)
                verticalLineToRelative(2.74f)
                curveToRelative(0.0f, 0.27f, 0.18f, 0.58f, 0.69f, 0.48f)
                curveTo(19.13f, 20.17f, 22.0f, 16.42f, 22.0f, 12.0f)
                arcTo(10.0f, 10.0f, 0.0f, false, false, 12.0f, 2.0f)
                close()
            }
        }.build()
    }
}

// Helper to get string resource ID dynamically from a string name
@Composable
fun getStringResourceId(name: String): Int {
    return LocalContext.current.resources.getIdentifier(
        name, "string", LocalContext.current.packageName
    )
}
