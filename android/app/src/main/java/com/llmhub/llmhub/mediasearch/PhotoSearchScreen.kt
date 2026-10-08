package com.llmhub.llmhub.mediasearch

import android.Manifest
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.llmhub.llmhub.R
import kotlinx.coroutines.launch

private fun photoPermissions(): Array<String> = when {
    Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE ->
        arrayOf(Manifest.permission.READ_MEDIA_IMAGES, Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
    Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU -> arrayOf(Manifest.permission.READ_MEDIA_IMAGES)
    else -> arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotoSearchScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModelDownload: () -> Unit,
    viewModel: PhotoSearchViewModel = viewModel()
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    val models by viewModel.downloadedModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val modelError by viewModel.modelError.collectAsState()
    val source by viewModel.source.collectAsState()
    val assets by viewModel.assets.collectAsState()
    val progress by viewModel.progress.collectAsState()
    val isPaused by viewModel.isPaused.collectAsState()
    val query by viewModel.query.collectAsState()
    val results by viewModel.results.collectAsState()
    val isSearching by viewModel.isSearching.collectAsState()
    val similarTo by viewModel.similarTo.collectAsState()

    var showSettings by remember { mutableStateOf(false) }
    var viewing by remember { mutableStateOf<PhotoAsset?>(null) }

    // Model downloads and permission changes happen outside this screen.
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) { viewModel.refresh() }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        if (grants.values.any { it }) {
            viewModel.useAllPhotos()
        } else {
            scope.launch { snackbarHostState.showSnackbar(context.getString(R.string.photo_search_permission_denied)) }
        }
    }
    val pickerLauncher = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { uris ->
        viewModel.addSelectedPhotos(uris)
    }
    val requestAllPhotos = { permissionLauncher.launch(photoPermissions()) }
    val pickPhotos = { pickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }

    if (showSettings) {
        MediaSearchSettingsSheet(
            models = models,
            selectedModel = selectedModel,
            isLoadingModel = isLoadingModel,
            isModelLoaded = isModelLoaded,
            onModelSelected = viewModel::selectModel,
            onLoadModel = viewModel::loadModel,
            onUnloadModel = viewModel::unloadModel,
            libraryCountText = stringResource(R.string.photo_search_count, progress.processed, assets.size),
            libraryActions = {
                OutlinedButton(onClick = pickPhotos, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                    Icon(Icons.Default.AddPhotoAlternate, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(stringResource(R.string.photo_search_select_photos))
                }
                if (source != PhotoSource.ALL_PHOTOS) {
                    OutlinedButton(onClick = requestAllPhotos, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                        Icon(Icons.Default.PhotoLibrary, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.photo_search_all_photos))
                    }
                }
            },
            onClearAll = {
                viewModel.clearAll()
                showSettings = false
            },
            onDismiss = { showSettings = false }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.ImageSearch, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(end = 8.dp))
                        Text(stringResource(R.string.feature_photo_search), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    if (models.isNotEmpty()) {
                        IconButton(onClick = { showSettings = true }) {
                            Icon(Icons.Default.Tune, contentDescription = stringResource(R.string.feature_settings_title))
                        }
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { padding ->
        val contentModifier = Modifier.fillMaxSize().padding(padding)
        when {
            models.isEmpty() -> MediaSearchDownloadGate(Icons.Default.ImageSearch, onNavigateToModelDownload, contentModifier)
            source == PhotoSource.NONE -> MediaSearchOnboarding(
                icon = Icons.Default.ImageSearch,
                title = stringResource(R.string.photo_search_onboarding_title),
                description = stringResource(R.string.photo_search_onboarding_desc),
                primaryLabel = stringResource(R.string.photo_search_all_photos),
                onPrimary = requestAllPhotos,
                secondaryLabel = stringResource(R.string.photo_search_select_photos),
                onSecondary = pickPhotos,
                modifier = contentModifier
            )
            else -> {
                val shown = results?.map { it.asset } ?: assets
                LazyVerticalGrid(
                    columns = GridCells.Adaptive(104.dp),
                    modifier = contentModifier.imePadding(),
                    contentPadding = PaddingValues(16.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    item(span = { GridItemSpan(maxLineSpan) }) {
                        Column(verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.padding(bottom = 8.dp)) {
                            MediaSearchField(query, viewModel::onQueryChange, stringResource(R.string.photo_search_hint), isSearching)
                            MediaSearchStatusCard(
                                isLoadingModel = isLoadingModel,
                                modelError = modelError,
                                progress = progress,
                                isPaused = isPaused,
                                onPause = viewModel::pauseIndexing,
                                onResume = viewModel::resumeIndexing,
                                onRetryModel = viewModel::loadModel
                            )
                            similarTo?.let { seed ->
                                SimilarToHeader(seed, onClear = viewModel::clearSearch)
                            }
                            if (results != null && !progress.isComplete) {
                                Text(
                                    stringResource(R.string.media_search_incomplete_warning),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            if (source == PhotoSource.ALL_PHOTOS && assets.isEmpty()) {
                                AccessNeededRow(onGrant = requestAllPhotos, onPick = pickPhotos)
                            }
                            if (results?.isEmpty() == true && !isSearching) {
                                Text(stringResource(R.string.media_search_no_results), style = MaterialTheme.typography.bodyMedium)
                            }
                        }
                    }
                    items(shown, key = { it.id }) { asset ->
                        AsyncImage(
                            model = ImageRequest.Builder(context).data(asset.uri).size(320).crossfade(true).build(),
                            contentDescription = null,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .aspectRatio(1f)
                                .clip(RoundedCornerShape(8.dp))
                                .clickable { viewing = asset }
                        )
                    }
                }
            }
        }
    }

    viewing?.let { asset ->
        PhotoViewer(
            asset = asset,
            onFindSimilar = {
                viewing = null
                viewModel.findSimilar(asset)
            },
            onDismiss = { viewing = null }
        )
    }
}

@Composable
private fun SimilarToHeader(seed: PhotoAsset, onClear: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        AsyncImage(
            model = seed.uri,
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.size(48.dp).clip(RoundedCornerShape(8.dp))
        )
        Text(
            stringResource(R.string.photo_search_similar_results),
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.weight(1f)
        )
        IconButton(onClick = onClear) { Icon(Icons.Default.Close, contentDescription = stringResource(R.string.close)) }
    }
}

@Composable
private fun AccessNeededRow(onGrant: () -> Unit, onPick: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(stringResource(R.string.photo_search_permission_denied), style = MaterialTheme.typography.bodyMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = onGrant) { Text(stringResource(R.string.media_search_grant_access)) }
            OutlinedButton(onClick = onPick) { Text(stringResource(R.string.photo_search_select_photos)) }
        }
    }
}

@Composable
private fun PhotoViewer(asset: PhotoAsset, onFindSimilar: () -> Unit, onDismiss: () -> Unit) {
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(modifier = Modifier.fillMaxSize(), color = Color.Black) {
            Box {
                AsyncImage(
                    model = asset.uri,
                    contentDescription = null,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize()
                )
                IconButton(onClick = onDismiss, modifier = Modifier.align(Alignment.TopEnd).padding(16.dp)) {
                    Icon(Icons.Default.Close, contentDescription = stringResource(R.string.close), tint = Color.White)
                }
                Button(
                    onClick = onFindSimilar,
                    modifier = Modifier.align(Alignment.BottomCenter).padding(32.dp),
                    shape = RoundedCornerShape(24.dp)
                ) {
                    Icon(Icons.Default.ImageSearch, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(stringResource(R.string.photo_search_find_similar))
                }
            }
        }
    }
}
