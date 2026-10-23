package com.llmhub.llmhub.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.CoroutineExceptionHandler
import android.os.PowerManager
import androidx.core.content.ContextCompat
import com.llmhub.llmhub.R
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelDownloader
import com.llmhub.llmhub.data.DownloadStatus
import com.llmhub.llmhub.data.localFileName

class ModelDownloadService : Service() {
    private val job = Job()
    private val scope = CoroutineScope(Dispatchers.IO + job)
    private var notificationManager: NotificationManager? = null
    private val channelId = "model_download_channel"
    private var notificationIdCounter = 1
    private val activeDownloads = mutableMapOf<String, Job>()

    private var wakeLock: PowerManager.WakeLock? = null

    companion object {
        private const val MANAGED_NOTIFICATION_ID = 41
        private const val ACTION_MANAGED_DOWNLOAD = "com.llmhub.llmhub.MANAGED_DOWNLOAD"
        // UI lifetimes must not own network transfers. All access is on Main.
        private val managedScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate +
            CoroutineExceptionHandler { _, error -> Log.e("ModelDownloadService", "Download failed", error) })
        private val managedJobs = mutableMapOf<String, Job>()
        private val snapshots = kotlinx.coroutines.flow.MutableStateFlow<Map<String, LLMModel>>(emptyMap())
        val downloadStates: kotlinx.coroutines.flow.StateFlow<Map<String, LLMModel>> = snapshots
        fun publishState(model: LLMModel) { snapshots.value = snapshots.value + (model.name to model) }
        fun isDownloading(name: String): Boolean = managedJobs[name]?.let { !it.isCompleted && !it.isCancelled } == true
        private var runningService: ModelDownloadService? = null

        fun launchDownload(context: Context, name: String, work: suspend () -> Unit): Job {
            managedJobs[name]?.takeUnless { it.isCancelled || it.isCompleted }?.let { return it }
            val task = managedScope.launch(start = CoroutineStart.LAZY) {
                try { work() } finally {
                    if (managedJobs[name] === coroutineContext[Job]) {
                        managedJobs.remove(name)
                        runningService?.finishManagedDownload(name)
                    }
                }
            }
            managedJobs[name] = task
            try {
                ContextCompat.startForegroundService(context, Intent(context, ModelDownloadService::class.java).apply {
                    action = ACTION_MANAGED_DOWNLOAD
                    putExtra(EXTRA_MODEL_NAME, name)
                })
            } catch (error: Exception) {
                managedJobs.remove(name)
                task.cancel()
                throw error
            }
            return task
        }

        fun pauseManaged(name: String) { managedJobs[name]?.cancel() }

        // Last posted (time, percent) per download. Every progress update posted a notification, a
        // binder call into system_server, on the main thread; under load they queued up into an ANR
        // (NotificationManager.notify from reportProgress, Play vitals). Post only when the percent
        // changes, at most twice a second.
        private val lastProgressPost = java.util.concurrent.ConcurrentHashMap<String, Pair<Long, Int>>()

        fun reportProgress(name: String, status: DownloadStatus) {
            val percent = if (status.totalBytes > 0) (status.downloadedBytes * 100 / status.totalBytes).toInt() else -1
            val now = android.os.SystemClock.elapsedRealtime()
            val last = lastProgressPost[name]
            if (last != null && percent < 100 && (last.second == percent || now - last.first < 500L)) return
            lastProgressPost[name] = now to percent
            runningService?.showNotification(name, status)
        }

        const val ACTION_DOWNLOAD_PROGRESS = "com.llmhub.llmhub.DOWNLOAD_PROGRESS"
        const val ACTION_DOWNLOAD_COMPLETED = "com.llmhub.llmhub.DOWNLOAD_COMPLETED"
        const val ACTION_DOWNLOAD_ERROR = "com.llmhub.llmhub.DOWNLOAD_ERROR"
        const val ACTION_PAUSE_DOWNLOAD = "com.llmhub.llmhub.PAUSE_DOWNLOAD"
        const val EXTRA_MODEL_NAME = "model_name"
        const val EXTRA_DOWNLOADED_BYTES = "downloaded_bytes"
        const val EXTRA_TOTAL_BYTES = "total_bytes"
        const val EXTRA_DOWNLOAD_SPEED = "download_speed"
        const val EXTRA_ERROR_MESSAGE = "error_message"
        const val ACTION_CANCEL_DOWNLOAD = "com.llmhub.llmhub.CANCEL_DOWNLOAD"
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        createNotificationChannel()
        runningService = this
    }

    private fun ensureAwake() {
        if (wakeLock?.isHeld != true) {
            wakeLock = (getSystemService(POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "LLMHub:ModelDownload").apply {
                    setReferenceCounted(false)
                    acquire(6 * 60 * 60 * 1000L)
                }
        }
    }

    private fun finishManagedDownload(name: String) {
        notificationManager?.cancel(getNotificationId(name))
        if (managedJobs.isEmpty() && activeDownloads.isEmpty()) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            if (wakeLock?.isHeld == true) wakeLock?.release()
            stopSelf()
        } else if (managedJobs.isNotEmpty()) {
            notificationManager?.notify(MANAGED_NOTIFICATION_ID, createInitialNotification(managedJobs.keys.first()))
        }
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        // Android limits dataSync services. Preserve partial files for Resume.
        managedJobs.values.toList().forEach { it.cancel() }
        activeDownloads.values.toList().forEach { it.cancel() }
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_MANAGED_DOWNLOAD) {
            val name = intent.getStringExtra(EXTRA_MODEL_NAME) ?: return START_NOT_STICKY
            startForeground(MANAGED_NOTIFICATION_ID, createInitialNotification(name))
            ensureAwake()
            managedJobs[name]?.start()
            if (managedJobs.isEmpty()) finishManagedDownload(name)
            return START_NOT_STICKY
        }
        when (intent?.action) {
            ACTION_CANCEL_DOWNLOAD -> {
                val modelName = intent.getStringExtra(EXTRA_MODEL_NAME) ?: return START_NOT_STICKY
                cancelDownload(modelName)
                return START_NOT_STICKY
            }
            ACTION_PAUSE_DOWNLOAD -> {
                val modelName = intent.getStringExtra(EXTRA_MODEL_NAME) ?: return START_NOT_STICKY
                pauseDownload(modelName)
                return START_NOT_STICKY
            }
        }

        val modelName = intent?.getStringExtra("modelName") ?: return START_NOT_STICKY
        val modelDescription = intent.getStringExtra("modelDescription") ?: ""
        val modelUrl = intent.getStringExtra("modelUrl") ?: return START_NOT_STICKY
        val modelSize = intent.getLongExtra("modelSize", -1L)
        val modelCategory = intent.getStringExtra("modelCategory") ?: "unknown"
        val modelSource = intent.getStringExtra("modelSource") ?: ""
        val supportsVision = intent.getBooleanExtra("supportsVision", false)
        val supportsGpu = intent.getBooleanExtra("supportsGpu", false)
        val minRamGB = intent.getIntExtra("minRamGB", 4)
        val recommendedRamGB = intent.getIntExtra("recommendedRamGB", 8)
        val requirements = com.llmhub.llmhub.data.ModelRequirements(minRamGB, recommendedRamGB)
        val hfToken = intent.getStringExtra("hfToken")
        val client = io.ktor.client.HttpClient() // Use a real client
        val model = LLMModel(
            name = modelName,
            description = modelDescription,
            url = modelUrl,
            category = modelCategory,
            sizeBytes = modelSize,
            source = modelSource,
            supportsVision = supportsVision,
            supportsGpu = supportsGpu,
            requirements = requirements
        )
        val downloader = ModelDownloader(client = client, context = applicationContext, hfToken = hfToken)

        // Check if this model is already being downloaded
        if (activeDownloads.containsKey(modelName)) {
            Log.w("ModelDownloadService", "Model $modelName is already being downloaded")
            return START_NOT_STICKY
        }

        // Start foreground service with initial notification if this is the first download
        if (activeDownloads.isEmpty() && managedJobs.isEmpty()) {
            val initialNotification = createInitialNotification(modelName)
            startForeground(getNotificationId(modelName), initialNotification)
        }

        ensureAwake()
        val downloadJob = scope.launch {
            try {
                downloader.downloadModel(model).collect { status ->
                    showNotification(modelName, status)
                    sendProgressBroadcast(modelName, status)
                }
                sendCompletedBroadcast(modelName)
                activeDownloads.remove(modelName)
                
                // Stop foreground service if no more active downloads
                if (activeDownloads.isEmpty() && managedJobs.isEmpty()) {
                    stopForeground(true)
                    stopSelf()
                }
            } catch (e: Exception) {
                showNotification(modelName, null, error = e.message)
                sendErrorBroadcast(modelName, e.message ?: "Unknown error")
                activeDownloads.remove(modelName)
                
                // Stop foreground service if no more active downloads
                if (activeDownloads.isEmpty() && managedJobs.isEmpty()) {
                    stopForeground(true)
                    stopSelf()
                }
            }
        }
        
        activeDownloads[modelName] = downloadJob
        return START_STICKY
    }

    override fun onDestroy() {
        if (runningService === this) runningService = null
        for ((name, task) in managedJobs.toMap()) {
            // A new start request may already be queued while this instance is
            // stopping. Leave its lazy job for the next service instance.
            if (task.isActive || task.isCancelled || task.isCompleted) {
                task.cancel()
                if (managedJobs[name] === task) managedJobs.remove(name)
            }
        }
        if (wakeLock?.isHeld == true) wakeLock?.release()
        wakeLock = null
        activeDownloads.values.forEach { it.cancel() }
        activeDownloads.clear()
        job.cancel()
        super.onDestroy()
    }

    private fun cancelDownload(modelName: String) {
        managedJobs[modelName]?.cancel()
        activeDownloads[modelName]?.cancel()
        activeDownloads.remove(modelName)
        
        // Delete partial file if exists
        val modelsDir = java.io.File(applicationContext.filesDir, "models")
        val model = com.llmhub.llmhub.data.LLMModel(
            name = modelName,
            description = "",
            url = "",
            category = "",
            sizeBytes = 0,
            source = "",
            supportsVision = false,
            supportsGpu = false,
            requirements = com.llmhub.llmhub.data.ModelRequirements(4, 8)
        )
        val primaryFile = java.io.File(modelsDir, model.localFileName())
        val legacyFile = java.io.File(modelsDir, "${modelName.replace(" ", "_")}.gguf")
        
        var deletedPrimary = false
        var deletedLegacy = false
        
        if (primaryFile.exists()) {
            deletedPrimary = primaryFile.delete()
            Log.d("ModelDownloadService", "[cancelDownload] Deleted primary file: $deletedPrimary, path: ${primaryFile.absolutePath}")
        }
        if (legacyFile.exists()) {
            deletedLegacy = legacyFile.delete()
            Log.d("ModelDownloadService", "[cancelDownload] Deleted legacy file: $deletedLegacy, path: ${legacyFile.absolutePath}")
        }
        
        // Verify files are actually gone
        val primaryExists = primaryFile.exists()
        val legacyExists = legacyFile.exists()
        Log.d("ModelDownloadService", "[cancelDownload] Post-deletion check - Primary exists: $primaryExists, Legacy exists: $legacyExists")
        
        // Stop foreground service if no more active downloads
        if (activeDownloads.isEmpty() && managedJobs.isEmpty()) {
            stopForeground(true)
            stopSelf()
        }
    }

    private fun pauseDownload(modelName: String) {
        managedJobs[modelName]?.cancel()
        activeDownloads[modelName]?.cancel()
        activeDownloads.remove(modelName)
        
        // Send pause broadcast
        val intent = Intent(ACTION_DOWNLOAD_ERROR).apply {
            putExtra(EXTRA_MODEL_NAME, modelName)
            putExtra(EXTRA_ERROR_MESSAGE, "Download paused")
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
        
        // Stop foreground service if no more active downloads
        if (activeDownloads.isEmpty() && managedJobs.isEmpty()) {
            stopForeground(true)
            stopSelf()
        }
    }

    private fun getNotificationId(modelName: String): Int {
        return modelName.hashCode().let { if (it > 0) it else -it } % 10000 + 1000
    }

    private fun showNotification(modelName: String, status: DownloadStatus?, error: String? = null) {
        val builder = NotificationCompat.Builder(this, channelId)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("${getString(R.string.downloading)} $modelName")
            .setOngoing(true)

        if (error != null) {
            builder.setContentText("Error: $error")
                .setOngoing(false)
        } else if (status != null) {
            val percent = if (status.totalBytes > 0 && status.downloadedBytes <= status.totalBytes) {
                (status.downloadedBytes * 100 / status.totalBytes).toInt()
            } else 0
            builder.setContentText("${status.downloadedBytes / (1024*1024)}MB / ${status.totalBytes / (1024*1024)}MB (${percent}%)")
                .setProgress(100, percent, false)
        } else {
            builder.setContentText(getString(R.string.downloading))
        }
        notificationManager?.notify(if (managedJobs.containsKey(modelName)) MANAGED_NOTIFICATION_ID else getNotificationId(modelName), builder.build())
    }

    private fun createInitialNotification(modelName: String): Notification {
        return NotificationCompat.Builder(this, channelId)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("${getString(R.string.downloading)} $modelName")
            .setContentText(getString(R.string.downloading))
            .setOngoing(true)
            .build()
    }

    private fun sendProgressBroadcast(modelName: String, status: DownloadStatus) {
        val intent = Intent(ACTION_DOWNLOAD_PROGRESS).apply {
            putExtra(EXTRA_MODEL_NAME, modelName)
            putExtra(EXTRA_DOWNLOADED_BYTES, status.downloadedBytes)
            putExtra(EXTRA_TOTAL_BYTES, status.totalBytes)
            putExtra(EXTRA_DOWNLOAD_SPEED, status.downloadSpeedBytesPerSec)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun sendCompletedBroadcast(modelName: String) {
        val intent = Intent(ACTION_DOWNLOAD_COMPLETED).apply {
            putExtra(EXTRA_MODEL_NAME, modelName)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun sendErrorBroadcast(modelName: String, error: String) {
        val intent = Intent(ACTION_DOWNLOAD_ERROR).apply {
            putExtra(EXTRA_MODEL_NAME, modelName)
            putExtra(EXTRA_ERROR_MESSAGE, error)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, getString(R.string.download_models), NotificationManager.IMPORTANCE_LOW)
            notificationManager?.createNotificationChannel(channel)
        }
    }
}
