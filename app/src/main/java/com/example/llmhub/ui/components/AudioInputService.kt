package com.llmhub.llmhub.ui.components

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AudioInputService(private val context: Context) {
    private var audioRecord: AudioRecord? = null
    private var outputFile: File? = null
    private var isRecording = false
    private var recordingThread: Thread? = null
    
    companion object {
        private const val TAG = "AudioInputService"
        private const val SAMPLE_RATE = 16000 // 16kHz for speech
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO // Mono channel
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT // 16-bit PCM
        private val BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
    }
    
    /**
     * Check if audio recording permission is granted
     */
    fun hasAudioPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context, 
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    /**
     * Start audio recording
     * @return true if recording started successfully, false otherwise
     */
    suspend fun startRecording(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!hasAudioPermission()) {
                Log.w(TAG, "Audio permission not granted")
                return@withContext false
            }
            
            if (isRecording) {
                Log.w(TAG, "Already recording")
                return@withContext false
            }
            
            // Create output file for WAV format
            outputFile = File.createTempFile("audio_", ".wav", context.cacheDir)
            
            // Initialize AudioRecord for mono WAV recording (required by MediaPipe)
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                BUFFER_SIZE
            )
            
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed")
                cleanup()
                return@withContext false
            }
            
            isRecording = true
            audioRecord?.startRecording()
            
            // Start recording thread
            recordingThread = Thread {
                writeAudioDataToFile()
            }
            recordingThread?.start()
            
            Log.d(TAG, "Started recording to: ${outputFile!!.absolutePath}")
            return@withContext true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
            cleanup()
            return@withContext false
        }
    }
    
    /**
     * Stop audio recording
     * @return ByteArray of recorded audio data, or null if failed
     */
    suspend fun stopRecording(): ByteArray? = withContext(Dispatchers.IO) {
        try {
            if (!isRecording) {
                Log.w(TAG, "Not currently recording")
                return@withContext null
            }
            
            isRecording = false
            
            // Stop AudioRecord
            audioRecord?.stop()
            
            // Wait for recording thread to finish
            recordingThread?.join(1000) // Wait up to 1 second
            
            // Release resources
            audioRecord?.release()
            audioRecord = null
            recordingThread = null
            
            val audioData = outputFile?.readBytes()
            Log.d(TAG, "Stopped recording, audio size: ${audioData?.size ?: 0} bytes")
            
            // Cleanup temp file
            outputFile?.delete()
            outputFile = null
            
            return@withContext audioData
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop recording", e)
            cleanup()
            return@withContext null
        }
    }
    
    /**
     * Cancel current recording
     */
    fun cancelRecording() {
        cleanup()
    }
    
    /**
     * Check if currently recording
     */
    fun isRecording(): Boolean = isRecording
    
    private fun cleanup() {
        try {
            isRecording = false
            audioRecord?.apply {
                if (state == AudioRecord.STATE_INITIALIZED) {
                    stop()
                }
                release()
            }
            recordingThread?.interrupt()
        } catch (e: Exception) {
            Log.w(TAG, "Error during cleanup", e)
        }
        
        audioRecord = null
        recordingThread = null
        outputFile?.delete()
        outputFile = null
    }
    
    /**
     * Write audio data to WAV file
     */
    private fun writeAudioDataToFile() {
        val data = ByteArray(BUFFER_SIZE)
        var output: FileOutputStream? = null
        
        try {
            output = FileOutputStream(outputFile)
            
            // Write WAV header (we'll update it later with actual data size)
            writeWavHeader(output, SAMPLE_RATE, 1, 16, 0) // 0 data size initially
            
            var totalDataSize = 0
            
            while (isRecording) {
                val bytesRead = audioRecord?.read(data, 0, BUFFER_SIZE) ?: 0
                if (bytesRead > 0) {
                    output.write(data, 0, bytesRead)
                    totalDataSize += bytesRead
                }
            }
            
            output.close()
            
            // Update WAV header with actual data size
            updateWavHeader(outputFile!!, totalDataSize)
            
        } catch (e: IOException) {
            Log.e(TAG, "Error writing audio data", e)
        } finally {
            output?.close()
        }
    }
    
    /**
     * Write WAV file header
     */
    private fun writeWavHeader(output: FileOutputStream, sampleRate: Int, channels: Int, bitsPerSample: Int, dataSize: Int) {
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        
        // RIFF chunk
        header.put("RIFF".toByteArray())
        header.putInt(36 + dataSize) // File size - 8
        header.put("WAVE".toByteArray())
        
        // Format chunk
        header.put("fmt ".toByteArray())
        header.putInt(16) // Subchunk1Size (PCM)
        header.putShort(1) // AudioFormat (PCM)
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(sampleRate * channels * bitsPerSample / 8) // ByteRate
        header.putShort((channels * bitsPerSample / 8).toShort()) // BlockAlign
        header.putShort(bitsPerSample.toShort())
        
        // Data chunk
        header.put("data".toByteArray())
        header.putInt(dataSize)
        
        output.write(header.array())
    }
    
    /**
     * Update WAV header with actual data size
     */
    private fun updateWavHeader(file: File, dataSize: Int) {
        try {
            val fileBytes = file.readBytes()
            val buffer = ByteBuffer.wrap(fileBytes).order(ByteOrder.LITTLE_ENDIAN)
            
            // Update file size at offset 4
            buffer.putInt(4, 36 + dataSize)
            
            // Update data size at offset 40
            buffer.putInt(40, dataSize)
            
            file.writeBytes(buffer.array())
        } catch (e: Exception) {
            Log.e(TAG, "Error updating WAV header", e)
        }
    }
}
