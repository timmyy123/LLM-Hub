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

    companion object {
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
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
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
        if (activeDownloads.isEmpty()) {
            val initialNotification = createInitialNotification(modelName)
            startForeground(getNotificationId(modelName), initialNotification)
        }

        val downloadJob = scope.launch {
            try {
                downloader.downloadModel(model).collect { status ->
                    showNotification(modelName, status)
                    sendProgressBroadcast(modelName, status)
                }
                sendCompletedBroadcast(modelName)
                activeDownloads.remove(modelName)
                
                // Stop foreground service if no more active downloads
                if (activeDownloads.isEmpty()) {
                    stopForeground(true)
                    stopSelf()
                }
            } catch (e: Exception) {
                showNotification(modelName, null, error = e.message)
                sendErrorBroadcast(modelName, e.message ?: "Unknown error")
                activeDownloads.remove(modelName)
                
                // Stop foreground service if no more active downloads
                if (activeDownloads.isEmpty()) {
                    stopForeground(true)
                    stopSelf()
                }
            }
        }
        
        activeDownloads[modelName] = downloadJob
        return START_STICKY
    }

    override fun onDestroy() {
        activeDownloads.values.forEach { it.cancel() }
        activeDownloads.clear()
        job.cancel()
        super.onDestroy()
    }

    private fun cancelDownload(modelName: String) {
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
        if (activeDownloads.isEmpty()) {
            stopForeground(true)
            stopSelf()
        }
    }

    private fun pauseDownload(modelName: String) {
        activeDownloads[modelName]?.cancel()
        activeDownloads.remove(modelName)
        
        // Send pause broadcast
        val intent = Intent(ACTION_DOWNLOAD_ERROR).apply {
            putExtra(EXTRA_MODEL_NAME, modelName)
            putExtra(EXTRA_ERROR_MESSAGE, "Download paused")
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
        
        // Stop foreground service if no more active downloads
        if (activeDownloads.isEmpty()) {
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
            .setContentTitle("Downloading $modelName")
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
            builder.setContentText("Starting download...")
        }
        notificationManager?.notify(getNotificationId(modelName), builder.build())
    }

    private fun createInitialNotification(modelName: String): Notification {
        return NotificationCompat.Builder(this, channelId)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Downloading $modelName")
            .setContentText("Starting download...")
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
            val channel = NotificationChannel(channelId, "Model Downloads", NotificationManager.IMPORTANCE_LOW)
            notificationManager?.createNotificationChannel(channel)
        }
    }
}
