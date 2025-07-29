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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelDownloader
import com.llmhub.llmhub.data.DownloadStatus

class ModelDownloadService : Service() {
    private val job = Job()
    private val scope = CoroutineScope(Dispatchers.IO + job)
    private var notificationManager: NotificationManager? = null
    private val channelId = "model_download_channel"
    private val notificationId = 1

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
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

        scope.launch {
            try {
                downloader.downloadModel(model).collect { status ->
                    showNotification(modelName, status)
                }
                stopSelf()
            } catch (e: Exception) {
                showNotification(modelName, null, error = e.message)
                stopSelf()
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        job.cancel()
        super.onDestroy()
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
        notificationManager?.notify(notificationId, builder.build())
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "Model Downloads", NotificationManager.IMPORTANCE_LOW)
            notificationManager?.createNotificationChannel(channel)
        }
    }
}
