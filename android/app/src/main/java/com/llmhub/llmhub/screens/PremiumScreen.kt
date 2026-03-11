package com.llmhub.llmhub.screens

import android.app.Activity
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.llmhub.llmhub.LlmHubApplication
import com.llmhub.llmhub.R
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PremiumScreen(
    onNavigateBack: () -> Unit
) {
    val context = LocalContext.current
    val activity = context as? Activity
    val billingManager = (context.applicationContext as LlmHubApplication).billingManager
    val coroutineScope = rememberCoroutineScope()

    val isPremium by billingManager.isPremium.collectAsState()
    val productPrice by billingManager.productPrice.collectAsState()

    var isRestoring by remember { mutableStateOf(false) }
    var restoreMessage by remember { mutableStateOf<String?>(null) }
    val snackbarHostState = remember { SnackbarHostState() }

    // Pulse animation for the crown icon
    val infiniteTransition = rememberInfiniteTransition(label = "crown_pulse")
    val crownScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.12f,
        animationSpec = infiniteRepeatable(
            animation = tween(900, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "crown_scale"
    )

    LaunchedEffect(restoreMessage) {
        restoreMessage?.let {
            snackbarHostState.showSnackbar(it)
            restoreMessage = null
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        topBar = {
            TopAppBar(
                title = {},
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.Close, contentDescription = stringResource(R.string.close))
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Transparent
                )
            )
        },
        containerColor = Color.Transparent
    ) { paddingValues ->

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        listOf(Color(0xFF1A0533), Color(0xFF0D1B4B))
                    )
                )
                .padding(paddingValues)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 24.dp, vertical = 16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {

                Spacer(modifier = Modifier.height(8.dp))

                // Animated crown icon
                Box(
                    modifier = Modifier
                        .size(96.dp)
                        .scale(crownScale)
                        .clip(CircleShape)
                        .background(
                            Brush.radialGradient(
                                listOf(Color(0xFFFFD700), Color(0xFFFFA500))
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.WorkspacePremium,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(56.dp)
                    )
                }

                Spacer(modifier = Modifier.height(20.dp))

                if (isPremium) {
                    // ── Already premium ─────────────────────────────────────────────
                    Text(
                        text = stringResource(R.string.premium_active_title),
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.ExtraBold,
                        color = Color(0xFFFFD700),
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = stringResource(R.string.premium_active_subtitle),
                        style = MaterialTheme.typography.bodyLarge,
                        color = Color.White.copy(alpha = 0.85f),
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(32.dp))
                    FilledTonalButton(
                        onClick = onNavigateBack,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(stringResource(R.string.close))
                    }
                } else {
                    // ── Upgrade flow ────────────────────────────────────────────────
                    Text(
                        text = stringResource(R.string.premium_title),
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.ExtraBold,
                        color = Color(0xFFFFD700),
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = stringResource(R.string.premium_subtitle),
                        style = MaterialTheme.typography.bodyLarge,
                        color = Color.White.copy(alpha = 0.75f),
                        textAlign = TextAlign.Center
                    )

                    Spacer(modifier = Modifier.height(28.dp))

                    // Feature list card
                    Surface(
                        shape = RoundedCornerShape(20.dp),
                        color = Color.White.copy(alpha = 0.08f),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(20.dp)) {
                            PremiumFeatureRow(
                                icon = Icons.Default.Block,
                                tint = Color(0xFFFF7043),
                                text = stringResource(R.string.premium_feature_no_ads)
                            )
                            PremiumFeatureRow(
                                icon = Icons.Default.Code,
                                tint = Color(0xFF7C4DFF),
                                text = stringResource(R.string.premium_feature_vibe_coder)
                            )
                            PremiumFeatureRow(
                                icon = Icons.Default.Palette,
                                tint = Color(0xFF2196F3),
                                text = stringResource(R.string.premium_feature_image_gen)
                            )
                            PremiumFeatureRow(
                                icon = Icons.Default.VolumeUp,
                                tint = Color(0xFF4CAF50),
                                text = stringResource(R.string.premium_feature_tts)
                            )
                            PremiumFeatureRow(
                                icon = Icons.Default.FileUpload,
                                tint = Color(0xFFFFC107),
                                text = stringResource(R.string.premium_feature_import_models)
                            )
                            PremiumFeatureRow(
                                icon = Icons.Default.Search,
                                tint = Color(0xFF29B6F6),
                                text = stringResource(R.string.premium_feature_web_search)
                            )
                            PremiumFeatureRow(
                                icon = Icons.Default.Psychology,
                                tint = Color(0xFF9C27B0),
                                text = stringResource(R.string.premium_feature_memory)
                            )
                            PremiumFeatureRow(
                                icon = Icons.Default.AutoAwesome,
                                tint = Color(0xFF00BCD4),
                                text = stringResource(R.string.premium_feature_future),
                                isLast = true
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(24.dp))

                    // Purchase button
                    Button(
                        onClick = { activity?.let { billingManager.launchPurchaseFlow(it) } },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(56.dp),
                        shape = RoundedCornerShape(16.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFFFFD700),
                            contentColor = Color(0xFF1A0533)
                        )
                    ) {
                        Icon(
                            Icons.Default.WorkspacePremium,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = if (productPrice != null)
                                stringResource(R.string.premium_button_unlock) + "  —  $productPrice"
                            else
                                stringResource(R.string.premium_price_loading),
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    // Restore purchases
                    TextButton(
                        onClick = {
                            isRestoring = true
                            coroutineScope.launch {
                                billingManager.restorePurchases()
                                isRestoring = false
                                restoreMessage = if (billingManager.isPremium.value)
                                    context.getString(R.string.premium_restore_success)
                                else
                                    context.getString(R.string.premium_restore_nothing)
                            }
                        },
                        enabled = !isRestoring
                    ) {
                        if (isRestoring) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(16.dp),
                                strokeWidth = 2.dp,
                                color = Color.White.copy(alpha = 0.7f)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                        }
                        Text(
                            stringResource(R.string.premium_button_restore),
                            color = Color.White.copy(alpha = 0.7f)
                        )
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    TextButton(onClick = onNavigateBack) {
                        Text(
                            stringResource(R.string.premium_button_later),
                            color = Color.White.copy(alpha = 0.45f),
                            fontSize = 13.sp
                        )
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    Text(
                        text = stringResource(R.string.premium_payment_note),
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.White.copy(alpha = 0.4f),
                        textAlign = TextAlign.Center
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    }
}

@Composable
private fun PremiumFeatureRow(
    icon: ImageVector,
    tint: Color,
    text: String,
    isLast: Boolean = false
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = if (isLast) 0.dp else 14.dp)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = tint,
            modifier = Modifier.size(22.dp)
        )
        Spacer(modifier = Modifier.width(14.dp))
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = Color.White.copy(alpha = 0.9f),
            fontWeight = FontWeight.Medium
        )
    }
}
