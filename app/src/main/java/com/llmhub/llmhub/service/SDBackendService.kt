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
import com.llmhub.llmhub.R
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Background service for Stable Diffusion backend
 * Based on local-dream's BackendService architecture
 * 
 * Runs libstable_diffusion_core.so as a native process that serves
 * an HTTP API on localhost:8081 for image generation.
 */
class SDBackendService : Service() {
    
    private val TAG = "SDBackendService"
    private val CHANNEL_ID = "sd_backend_channel"
    private val NOTIFICATION_ID = 2001
    private val EXECUTABLE_NAME = "libstable_diffusion_core.so"
    private val RUNTIME_DIR = "runtime_libs"
    
    private var backendProcess: Process? = null
    private lateinit var runtimeDir: File
    private var selectedModelPath: String? = null
    
    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "SDBackendService created")
        createNotificationChannel()
        prepareRuntimeDir()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "SDBackendService started: action=${intent?.action}")
        
        startForeground(NOTIFICATION_ID, createNotification("Initializing..."))
        
        when (intent?.action) {
            ACTION_START -> {
                selectedModelPath = intent.getStringExtra(EXTRA_MODEL_PATH)
                if (startBackend()) {
                    updateNotification("Backend Running")
                } else {
                    updateNotification("Backend Failed")
                    stopSelf()
                }
            }
            ACTION_STOP -> stopBackend()
        }
        
        return START_NOT_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    override fun onDestroy() {
        Log.i(TAG, "SDBackendService destroyed")
        stopBackend()
        super.onDestroy()
    }
    
    /**
     * Extract QNN runtime libraries from assets to filesDir
     * This must be done because assets are read-only and libraries need execute permission
     */
    private fun prepareRuntimeDir() {
        try {
            runtimeDir = File(filesDir, RUNTIME_DIR).apply {
                if (!exists()) {
                    mkdirs()
                }
            }
            
            // Extract QNN libraries from assets
            val qnnLibs = assets.list("qnnlibs") ?: emptyArray()
            Log.i(TAG, "Found ${qnnLibs.size} QNN libraries in assets")
            
            qnnLibs.forEach { fileName ->
                val targetFile = File(runtimeDir, fileName)
                
                // Check if we need to copy (file doesn't exist or size mismatch)
                val needsCopy = !targetFile.exists() || run {
                    val assetSize = assets.open("qnnlibs/$fileName").use { it.available().toLong() }
                    targetFile.length() != assetSize
                }
                
                if (needsCopy) {
                    assets.open("qnnlibs/$fileName").use { input ->
                        targetFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                    
                    // Set permissions
                    targetFile.setReadable(true, true)
                    targetFile.setExecutable(true, true)
                    
                    Log.d(TAG, "Extracted: $fileName")
                }
            }
            
            runtimeDir.setReadable(true, true)
            runtimeDir.setExecutable(true, true)
            
            Log.i(TAG, "Runtime directory prepared: ${runtimeDir.absolutePath}")
            
        } catch (e: IOException) {
            Log.e(TAG, "Failed to prepare runtime directory", e)
            throw RuntimeException("Failed to prepare runtime directory", e)
        }
    }
    
    /**
     * Start the native backend process
     * Runs libstable_diffusion_core.so with appropriate model files and configuration
     */
    private fun startBackend(): Boolean {
        try {
            val modelDir = if (selectedModelPath != null) {
                File(selectedModelPath!!)
            } else {
                File(filesDir, "sd_models")
            }
            
            if (!modelDir.exists()) {
                Log.e(TAG, "Model directory not found: ${modelDir.absolutePath}")
                return false
            }
            
            // Detect model type
            val modelType = detectModelType(modelDir)
            Log.i(TAG, "Starting $modelType backend")
            
            // Get native library directory
            val nativeDir = applicationInfo.nativeLibraryDir
            val executable = File(nativeDir, EXECUTABLE_NAME)
            
            if (!executable.exists()) {
                Log.e(TAG, "Executable not found: ${executable.absolutePath}")
                return false
            }
            
            // Build command based on model type
            val command = buildCommand(executable, modelDir, modelType)
            
            // Setup environment
            val env = buildEnvironment()
            
            Log.d(TAG, "Command: ${command.joinToString(" ")}")
            Log.d(TAG, "LD_LIBRARY_PATH=${env["LD_LIBRARY_PATH"]}")
            Log.d(TAG, "DSP_LIBRARY_PATH=${env["DSP_LIBRARY_PATH"]}")
            
            // Start process
            val processBuilder = ProcessBuilder(command).apply {
                directory(File(nativeDir))
                redirectErrorStream(true)
                environment().putAll(env)
            }
            
            backendProcess = processBuilder.start()
            
            // Start output monitor thread
            startMonitorThread()
            
            Log.i(TAG, "Backend process started successfully")
            return true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start backend", e)
            return false
        }
    }
    
    /**
     * Build command line arguments based on model type
     */
    private fun buildCommand(executable: File, modelDir: File, modelType: String): List<String> {
        val actualDir = findActualModelDir(modelDir)
        
        // Find clip file - for MNN models, always pass "clip.mnn" path
        // The native backend checks for clip.mnn suffix and auto-upgrades to clip_v2.mnn if exists
        // This triggers automatic loading of token_emb.bin and pos_emb.bin
        val clipFile = when {
            File(actualDir, "clip.bin").exists() -> File(actualDir, "clip.bin")
            // If clip_v2.mnn exists, pass clip.mnn path - backend auto-upgrades and loads embeddings
            File(actualDir, "clip_v2.mnn").exists() -> File(actualDir, "clip.mnn")
            File(actualDir, "clip.mnn").exists() -> File(actualDir, "clip.mnn")
            else -> File(actualDir, "clip.bin") // fallback
        }
        
        // Check if using MNN CLIP (hybrid mode: MNN CLIP + QNN UNet/VAE)
        val isClipMnn = clipFile.name.endsWith(".mnn")
        
        return if (modelType == "qnn") {
            // NPU backend with Qualcomm QNN
            val command = mutableListOf(
                executable.absolutePath,
                "--clip", clipFile.absolutePath,
                "--unet", File(actualDir, "unet.bin").absolutePath,
                "--vae_decoder", File(actualDir, "vae_decoder.bin").absolutePath,
                "--tokenizer", File(actualDir, "tokenizer.json").absolutePath,
                "--backend", File(runtimeDir, "libQnnHtp.so").absolutePath,
                "--system_library", File(runtimeDir, "libQnnSystem.so").absolutePath,
                "--port", "8081",
                "--text_embedding_size", "768"
            )
            
            // Add --use_cpu_clip flag for hybrid mode (MNN CLIP + QNN UNet/VAE)
            // The backend will auto-load token_emb.bin and pos_emb.bin when it detects clip_v2.mnn
            if (isClipMnn) {
                command.add("--use_cpu_clip")
                Log.i(TAG, "Using hybrid mode: MNN CLIP + QNN UNet/VAE")
            }
            
            command
        } else {
            // CPU backend with MNN
            listOf(
                executable.absolutePath,
                "--clip", clipFile.absolutePath,
                "--unet", File(actualDir, "unet.mnn").absolutePath,
                "--vae_decoder", File(actualDir, "vae_decoder.mnn").absolutePath,
                "--tokenizer", File(actualDir, "tokenizer.json").absolutePath,
                "--port", "8081",
                "--text_embedding_size", "768",
                "--cpu"
            )
        }
    }
    
    /**
     * Build environment variables for the backend process
     */
    private fun buildEnvironment(): MutableMap<String, String> {
        val env = mutableMapOf<String, String>()
        
        val systemLibPaths = listOf(
            runtimeDir.absolutePath,
            "/system/lib64",
            "/vendor/lib64",
            "/vendor/lib64/egl"
        ).joinToString(":")
        
        env["LD_LIBRARY_PATH"] = systemLibPaths
        env["DSP_LIBRARY_PATH"] = runtimeDir.absolutePath
        
        return env
    }
    
    /**
     * Detect which model type is available (QNN or MNN)
     */
    private fun detectModelType(modelDir: File): String {
        val actualDir = findActualModelDir(modelDir)
        return when {
            File(actualDir, "unet.bin").exists() -> "qnn"
            File(actualDir, "unet.mnn").exists() -> "mnn"
            else -> "unknown"
        }
    }
    
    /**
     * Find the actual directory containing model files
     * Some ZIPs extract to a subdirectory, so we need to search recursively
     */
    private fun findActualModelDir(baseDir: File): File {
        // Check if model files are in root directory
        if (File(baseDir, "unet.bin").exists() || File(baseDir, "unet.mnn").exists()) {
            return baseDir
        }
        
        // Search subdirectories recursively (up to 2 levels deep)
        fun searchDir(dir: File, depth: Int): File? {
            if (depth > 2) return null
            
            dir.listFiles()?.forEach { subDir ->
                if (subDir.isDirectory) {
                    // Check for QNN (.bin) or MNN (.mnn) model files
                    if (File(subDir, "unet.bin").exists() || File(subDir, "unet.mnn").exists()) {
                        Log.i(TAG, "Found model files in subdirectory: ${subDir.absolutePath}")
                        return subDir
                    }
                    // Recurse into subdirectories
                    searchDir(subDir, depth + 1)?.let { return it }
                }
            }
            return null
        }
        
        // Search and return found directory, or base directory as fallback
        return searchDir(baseDir, 0) ?: baseDir
    }
    
    /**
     * Monitor backend process output
     * Logs all stdout/stderr from the native process
     */
    private fun startMonitorThread() {
        Thread {
            try {
                backendProcess?.let { process ->
                    process.inputStream.bufferedReader().use { reader ->
                        var line: String?
                        while (reader.readLine().also { line = it } != null) {
                            Log.i(TAG, "Backend: $line")
                        }
                    }
                    
                    val exitCode = process.waitFor()
                    Log.w(TAG, "Backend process exited with code: $exitCode")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Monitor thread error", e)
            }
        }.apply {
            isDaemon = true
            name = "SD-Backend-Monitor"
            start()
        }
    }
    
    /**
     * Stop the backend process
     */
    private fun stopBackend() {
        Log.i(TAG, "Stopping backend")
        
        backendProcess?.let { process ->
            try {
                // Try graceful shutdown first
                process.destroy()
                
                // Wait up to 5 seconds for graceful shutdown
                if (!process.waitFor(5, TimeUnit.SECONDS)) {
                    Log.w(TAG, "Force killing backend process")
                    process.destroyForcibly()
                }
                
                Log.i(TAG, "Backend stopped with exit code: ${process.exitValue()}")
                
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping backend", e)
            } finally {
                backendProcess = null
            }
        }
        
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }
    
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "SD Backend Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Stable Diffusion inference backend"
                setShowBadge(false)
            }
            
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }
    
    private fun createNotification(contentText: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Image Generation")
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }
    
    private fun updateNotification(contentText: String) {
        val notification = createNotification(contentText)
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, notification)
    }
    
    companion object {
        const val ACTION_START = "com.llmhub.llmhub.SD_BACKEND_START"
        const val ACTION_STOP = "com.llmhub.llmhub.SD_BACKEND_STOP"
        const val EXTRA_MODEL_PATH = "model_path"
        
        fun start(context: Context, modelPath: String? = null) {
            val intent = Intent(context, SDBackendService::class.java).apply {
                action = ACTION_START
                if (modelPath != null) {
                    putExtra(EXTRA_MODEL_PATH, modelPath)
                }
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }
        
        fun stop(context: Context) {
            val intent = Intent(context, SDBackendService::class.java).apply {
                action = ACTION_STOP
            }
            context.startService(intent)
        }
    }
}
