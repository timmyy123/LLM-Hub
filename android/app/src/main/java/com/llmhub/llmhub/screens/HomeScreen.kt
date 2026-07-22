package com.llmhub.llmhub.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
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
import androidx.compose.ui.graphics.luminance
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
import androidx.compose.ui.platform.LocalContext
import com.llmhub.llmhub.ads.BannerAd
import com.llmhub.llmhub.data.ThemePreferences

data class FeatureCard(
    val title: Int,
    val description: Int,
    val icon: ImageVector,
    val gradient: Pair<Color, Color>,
    val darkColor: Color,
    val route: String
)


/** Routes that require a Premium subscription. */
private val PREMIUM_ROUTES = setOf("image_generator", "vibe_coder")

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onNavigateToFeature: (String) -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateToModels: () -> Unit,
    onNavigateToChatHistory: () -> Unit,
    onNavigateToPremium: () -> Unit = {},
    isPremium: Boolean = false
) {
    val uriHandler = LocalUriHandler.current
    val context = LocalContext.current
    val preferences = remember { ThemePreferences(context) }
    
    // Feature cards data
    val features = remember {
        listOf(
            FeatureCard(
                title = R.string.feature_ai_chat,
                description = R.string.feature_ai_chat_desc,
                icon = Icons.Filled.Chat,
                gradient = Pair(Color(0xFF667eea), Color(0xFF764ba2)),
                darkColor = Color(0xFF2B225E),
                route = "chat"
            ),
            FeatureCard(
                title = R.string.feature_writing_aid,
                description = R.string.feature_writing_aid_desc,
                icon = Icons.Filled.Edit,
                gradient = Pair(Color(0xFFf093fb), Color(0xFFf5576c)),
                darkColor = Color(0xFF7B2D4E),
                route = "writing_aid"
            ),
            FeatureCard(
                title = R.string.feature_translator,
                description = R.string.feature_translator_desc,
                icon = Icons.Filled.Language,
                gradient = Pair(Color(0xFF4facfe), Color(0xFF00f2fe)),
                darkColor = Color(0xFF1A4971),
                route = "translator"
            ),
            FeatureCard(
                title = R.string.feature_transcriber,
                description = R.string.feature_transcriber_desc,
                icon = Icons.Filled.Mic,
                gradient = Pair(Color(0xFF43e97b), Color(0xFF38f9d7)),
                darkColor = Color(0xFF1B5E3A),
                route = "transcriber"
            ),
            FeatureCard(
                title = R.string.feature_scam_detector,
                description = R.string.feature_scam_detector_desc,
                icon = Icons.Filled.Security,
                gradient = Pair(Color(0xFFfa709a), Color(0xFFfee140)),
                darkColor = Color(0xFF6E3028),
                route = "scam_detector"
            ),
            FeatureCard(
                title = R.string.feature_image_generator,
                description = R.string.feature_image_generator_desc,
                icon = Icons.Filled.Palette,
                gradient = Pair(Color(0xFF6a11cb), Color(0xFF2575fc)),
                darkColor = Color(0xFF2A1854),
                route = "image_generator"
            ),
            FeatureCard(
                title = R.string.feature_vibe_coder,
                description = R.string.feature_vibe_coder_desc,
                icon = Icons.Filled.Code,
                gradient = Pair(Color(0xFFf794a4), Color(0xFFfdd6bd)),
                darkColor = Color(0xFF4A2D3A),
                route = "vibe_coder"
            ),
            FeatureCard(
                title = R.string.feature_creator_generation,
                description = R.string.feature_creator_generation_desc,
                icon = Icons.Filled.AutoAwesome,
                gradient = Pair(Color(0xFF8EC5FC), Color(0xFFE0C3FC)),
                darkColor = Color(0xFF2E3D5C),
                route = "creator_generation"
            ),
            FeatureCard(
                title = R.string.feature_image_upscale,
                description = R.string.feature_image_upscale_desc,
                icon = Icons.Filled.AutoFixHigh,
                gradient = Pair(Color(0xFF11998e), Color(0xFF38ef7d)),
                darkColor = Color(0xFF14403A),
                route = "image_upscale"
            ),
            FeatureCard(
                title = R.string.feature_vibevoice,
                description = R.string.feature_vibevoice_desc,
                icon = Icons.Filled.GraphicEq,
                gradient = Pair(Color(0xFF7BC6CC), Color(0xFF8A82FB)),
                darkColor = Color(0xFF263F4A),
                route = "vibevoice"
            )
        )
    }

    val aiChatFeature = features.first { it.route == "chat" }
    val toolsFeatures = features.filter { it.route != "chat" }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
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

                    // Go Premium button (only for free users)
                    if (!isPremium) {
                        IconButton(onClick = onNavigateToPremium) {
                            Icon(
                                imageVector = Icons.Default.WorkspacePremium,
                                contentDescription = stringResource(R.string.premium_go_premium),
                                tint = Color(0xFFFFD700)
                            )
                        }
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
            val isLandscapeLayout = maxWidth > maxHeight
            val isTabletLayout = minOf(maxWidth, maxHeight) >= 600.dp
            val isPhoneLandscapeLayout = isLandscapeLayout && !isTabletLayout

            val isTabletPortrait = isTabletLayout && !isLandscapeLayout
            val isTabletLandscape = isTabletLayout && isLandscapeLayout

            val horizontalPadding = when {
                isPhoneLandscapeLayout -> 10.dp
                isTabletLayout -> 24.dp
                else -> 16.dp
            }
            val verticalPadding = when {
                isPhoneLandscapeLayout -> 8.dp
                isTabletLayout -> 20.dp
                else -> 16.dp
            }
            val sectionSpacing = when {
                isPhoneLandscapeLayout -> 8.dp
                else -> 14.dp
            }
            val rowSpacing = when {
                isPhoneLandscapeLayout -> 8.dp
                else -> 12.dp
            }

            val heroHeight = when {
                isPhoneLandscapeLayout -> 90.dp
                isTabletPortrait -> 224.dp
                isTabletLandscape -> 156.dp
                else -> 190.dp
            }

            val toolsColumns = if (isLandscapeLayout) 4 else 2

            val featureCardHeight = when {
                isPhoneLandscapeLayout -> 72.dp
                isTabletPortrait -> 118.dp
                isTabletLandscape -> 96.dp
                else -> 108.dp
            }

            val compactCards = isPhoneLandscapeLayout
            val sectionTitleStyle = if (isPhoneLandscapeLayout) {
                MaterialTheme.typography.titleLarge
            } else {
                MaterialTheme.typography.headlineSmall
            }

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = horizontalPadding, vertical = verticalPadding),
                verticalArrangement = Arrangement.spacedBy(sectionSpacing)
            ) {
                item {
                    HomeHeroCard(
                        feature = aiChatFeature,
                        cardHeight = heroHeight,
                        compact = compactCards,
                        isLocked = !isPremium && aiChatFeature.route in PREMIUM_ROUTES,
                        onClick = {
                            val isLocked = !isPremium && aiChatFeature.route in PREMIUM_ROUTES
                            if (isLocked) onNavigateToPremium() else onNavigateToFeature(aiChatFeature.route)
                        }
                    )
                }

                item {
                    Text(
                        text = stringResource(R.string.home_section_tools),
                        style = sectionTitleStyle,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }

                item {
                    Column(verticalArrangement = Arrangement.spacedBy(rowSpacing)) {
                        toolsFeatures.chunked(toolsColumns).forEach { rowFeatures ->
                            Row(horizontalArrangement = Arrangement.spacedBy(rowSpacing), modifier = Modifier.fillMaxWidth()) {
                                rowFeatures.forEach { feature ->
                                    val isLocked = !isPremium && feature.route in PREMIUM_ROUTES
                                    Box(modifier = Modifier.weight(1f)) {
                                        SmallFeatureCard(
                                            feature = feature,
                                            cardHeight = featureCardHeight,
                                            compact = compactCards,
                                            isLocked = isLocked,
                                            onClick = {
                                                if (isLocked) onNavigateToPremium() else onNavigateToFeature(feature.route)
                                            }
                                        )
                                    }
                                }
                                repeat((toolsColumns - rowFeatures.size).coerceAtLeast(0)) {
                                    Spacer(modifier = Modifier.weight(1f))
                                }
                            }
                        }
                    }
                }

                if (!isPremium && !isLandscapeLayout) {
                    item {
                        BannerAd(modifier = Modifier.fillMaxWidth())
                    }
                }

                item { Spacer(modifier = Modifier.height(6.dp)) }
            }
        }
    }
}

@Composable
private fun HomeHeroCard(
    feature: FeatureCard,
    cardHeight: Dp,
    compact: Boolean,
    isLocked: Boolean,
    onClick: () -> Unit
) {
    val isDark = MaterialTheme.colorScheme.surface.luminance() < 0.5f

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(cardHeight)
            .clickable { onClick() },
        shape = RoundedCornerShape(28.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 6.dp)
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    if (isDark) {
                        Brush.linearGradient(listOf(
                            feature.darkColor.copy(alpha = if (isLocked) 0.45f else 1f),
                            feature.darkColor.copy(alpha = if (isLocked) 0.45f else 1f)
                        ))
                    } else {
                        Brush.linearGradient(listOf(
                            feature.gradient.first.copy(alpha = if (isLocked) 0.45f else 1f),
                            feature.gradient.second.copy(alpha = if (isLocked) 0.45f else 1f)
                        ))
                    }
                )
                .padding(if (compact) 12.dp else 18.dp)
        ) {
            Column(
                modifier = Modifier.align(Alignment.CenterStart),
                verticalArrangement = Arrangement.spacedBy(if (compact) 6.dp else 10.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(if (compact) 8.dp else 12.dp)) {
                    Box(
                        modifier = Modifier
                            .size(if (compact) 42.dp else 54.dp)
                            .clip(RoundedCornerShape(14.dp))
                            .background(Color.White.copy(alpha = 0.28f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = feature.icon,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(if (compact) 24.dp else 30.dp)
                        )
                    }
                    Text(
                        text = stringResource(feature.title),
                        style = if (compact) MaterialTheme.typography.headlineSmall else MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.ExtraBold,
                        color = Color.White,
                        maxLines = 1
                    )
                }

                if (feature.route != "chat") {
                    Text(
                        text = stringResource(feature.description),
                        style = if (compact) MaterialTheme.typography.bodySmall else MaterialTheme.typography.bodyLarge,
                        color = Color.White.copy(alpha = 0.92f),
                        maxLines = if (compact) 1 else 2
                    )
                }

                Surface(
                    shape = RoundedCornerShape(24.dp),
                    color = Color.White.copy(alpha = 0.65f),
                    modifier = Modifier.widthIn(min = if (compact) 124.dp else 160.dp)
                ) {
                    Text(
                        text = stringResource(R.string.chat_now),
                        modifier = Modifier.padding(vertical = if (compact) 6.dp else 10.dp),
                        textAlign = TextAlign.Center,
                        style = if (compact) MaterialTheme.typography.titleSmall else MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF233E88)
                    )
                }
            }

            Icon(
                imageVector = Icons.Filled.ChatBubbleOutline,
                contentDescription = null,
                tint = Color.White.copy(alpha = 0.28f),
                modifier = Modifier
                    .align(Alignment.CenterEnd)
                    .size(if (compact) 72.dp else 120.dp)
            )

            if (isLocked) {
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .clip(RoundedCornerShape(bottomStart = 10.dp))
                        .background(Color(0xFFFFD700))
                        .padding(horizontal = 6.dp, vertical = 3.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = stringResource(R.string.premium_locked_feature),
                        tint = Color(0xFF1A0533),
                        modifier = Modifier.size(14.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun SmallFeatureCard(
    feature: FeatureCard,
    cardHeight: Dp,
    compact: Boolean,
    isLocked: Boolean,
    onClick: () -> Unit
) {
    val isDark = MaterialTheme.colorScheme.surface.luminance() < 0.5f

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(cardHeight)
            .clickable { onClick() },
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    if (isDark) {
                        Brush.linearGradient(listOf(
                            feature.darkColor.copy(alpha = if (isLocked) 0.45f else 1f),
                            feature.darkColor.copy(alpha = if (isLocked) 0.45f else 1f)
                        ))
                    } else {
                        Brush.linearGradient(listOf(
                            feature.gradient.first.copy(alpha = if (isLocked) 0.45f else 0.95f),
                            feature.gradient.second.copy(alpha = if (isLocked) 0.45f else 0.95f)
                        ))
                    }
                )
                .padding(if (compact) 8.dp else 12.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxHeight(),
                verticalArrangement = if (compact) Arrangement.spacedBy(4.dp) else Arrangement.spacedBy(8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(if (compact) 28.dp else 40.dp)
                        .clip(CircleShape)
                        .background(Color.White.copy(alpha = 0.25f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = feature.icon,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(if (compact) 16.dp else 22.dp)
                    )
                }

                Text(
                    text = stringResource(feature.title),
                    style = if (compact) MaterialTheme.typography.titleSmall else MaterialTheme.typography.titleLarge,
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1
                )
            }

            if (isLocked) {
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .clip(RoundedCornerShape(bottomStart = 10.dp))
                        .background(Color(0xFFFFD700))
                        .padding(horizontal = 6.dp, vertical = 3.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = stringResource(R.string.premium_locked_feature),
                        tint = Color(0xFF1A0533),
                        modifier = Modifier.size(14.dp)
                    )
                }
            }
        }
    }
}



@Composable
fun AnimatedFeatureCard(
    feature: FeatureCard,
    index: Int,
    cardHeight: Dp? = null,
    startTime: Long,
    isLocked: Boolean = false,
    onClick: () -> Unit
) {
    val isDark = MaterialTheme.colorScheme.surface.luminance() < 0.5f

    val initialDelay = index * 80L
    val timePassed = System.currentTimeMillis() - startTime

    var visible by remember {
        mutableStateOf(timePassed > initialDelay)
    }
    var isPressed by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        if (!visible) {
            val delayNeeded = (initialDelay - timePassed).coerceAtLeast(0L)
            if (delayNeeded > 0) delay(delayNeeded)
            visible = true
        }
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
                        if (isDark) {
                            Brush.linearGradient(listOf(
                                feature.darkColor.copy(alpha = if (isLocked) 0.45f else 1f),
                                feature.darkColor.copy(alpha = if (isLocked) 0.45f else 1f)
                            ))
                        } else {
                            Brush.linearGradient(listOf(
                                feature.gradient.first.copy(alpha = if (isLocked) 0.45f else 0.9f),
                                feature.gradient.second.copy(alpha = if (isLocked) 0.45f else 0.9f)
                            ))
                        }
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
                        text = stringResource(id = feature.title),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    // Description
                    Text(
                        text = stringResource(id = feature.description),
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.White.copy(alpha = 0.8f),
                        textAlign = TextAlign.Center,
                        lineHeight = 16.sp,
                        modifier = Modifier.padding(horizontal = 8.dp)
                    )
                }

                // Lock badge — top-right corner for premium-gated features
                if (isLocked) {
                    Box(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .clip(RoundedCornerShape(bottomStart = 10.dp))
                            .background(Color(0xFFFFD700))
                            .padding(horizontal = 6.dp, vertical = 3.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.Lock,
                            contentDescription = stringResource(R.string.premium_locked_feature),
                            tint = Color(0xFF1A0533),
                            modifier = Modifier.size(14.dp)
                        )
                    }
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
            addPath(
                pathData = androidx.compose.ui.graphics.vector.PathParser()
                    .parsePathString(
                        "M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.09-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2Z"
                    )
                    .toNodes(),
                fill = androidx.compose.ui.graphics.SolidColor(androidx.compose.ui.graphics.Color.Black)
            )
        }.build()
    }
}


