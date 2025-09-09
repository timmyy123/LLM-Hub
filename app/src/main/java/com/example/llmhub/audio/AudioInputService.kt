package com.llmhub.llmhub.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Service for recording and processing audio input for Gemma-3n models.
 * 
 * Audio Requirements for MediaPipe Gemma-3n models (from official docs):
 * - Format: WAV (mono channel)
 * - Must be mono channel formatted as .wav
 * - Compatible with Gemma-3n E2B and Gemma-3n E4B models
 * 
 * Following official Google AI Edge MediaPipe LLM Inference documentation:
 * https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android#audio_input
 */
interface AudioInputService {
    suspend fun hasAudioPermission(): Boolean
    suspend fun startRecording(): Flow<AudioRecordingState>
    suspend fun stopRecording(): ByteArray?
    suspend fun convertToWav(audioData: ByteArray): ByteArray
    fun cleanup()
}

sealed class AudioRecordingState {
    object Idle : AudioRecordingState()
    object Recording : AudioRecordingState()
    data class AudioLevel(val level: Float) : AudioRecordingState()
    data class Error(val message: String) : AudioRecordingState()
}

class MediaPipeAudioInputService(private val context: Context) : AudioInputService {
    
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val audioBuffer = ByteArrayOutputStream()
    
    companion object {
        private const val TAG = "AudioInputService"
        
        // Audio configuration for MediaPipe Gemma-3n (WAV mono 16kHz)
        private const val SAMPLE_RATE = 16000 // 16kHz - standard for speech recognition
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val BUFFER_SIZE_MULTIPLIER = 4 // Larger buffer for stability
    }
    
    override suspend fun hasAudioPermission(): Boolean {
        return ActivityCompat.checkSelfPermission(
            context,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    override suspend fun startRecording(): Flow<AudioRecordingState> = callbackFlow {
        if (!hasAudioPermission()) {
            trySend(AudioRecordingState.Error("Audio recording permission not granted"))
            close()
            return@callbackFlow
        }
        
        try {
            val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT) * BUFFER_SIZE_MULTIPLIER
            
            if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
                trySend(AudioRecordingState.Error("Failed to get audio buffer size"))
                close()
                return@callbackFlow
            }
            
            Log.d(TAG, "Creating AudioRecord with buffer size: $bufferSize")
            
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )
            
            val record = audioRecord
            if (record == null || record.state != AudioRecord.STATE_INITIALIZED) {
                trySend(AudioRecordingState.Error("Failed to initialize AudioRecord"))
                close()
                return@callbackFlow
            }
            
            // Clear previous recording data
            audioBuffer.reset()
            
            record.startRecording()
            isRecording = true
            
            trySend(AudioRecordingState.Recording)
            Log.d(TAG, "Started audio recording")
            
            val buffer = ByteArray(bufferSize / 4) // Smaller read chunks for responsiveness
            
            while (isRecording && !isClosedForSend) {
                val bytesRead = record.read(buffer, 0, buffer.size)
                
                if (bytesRead > 0) {
                    // Store audio data
                    audioBuffer.write(buffer, 0, bytesRead)
                    
                    // Calculate audio level for UI feedback
                    val audioLevel = calculateAudioLevel(buffer, bytesRead)
                    trySend(AudioRecordingState.AudioLevel(audioLevel))
                } else {
                    Log.w(TAG, "AudioRecord read returned: $bytesRead")
                }
            }
            
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception during recording", e)
            trySend(AudioRecordingState.Error("Audio recording permission denied"))
        } catch (e: Exception) {
            Log.e(TAG, "Error during recording", e)
            trySend(AudioRecordingState.Error("Recording failed: ${e.message}"))
        } finally {
            cleanup()
            trySend(AudioRecordingState.Idle)
            close()
        }
    }
    
    override suspend fun stopRecording(): ByteArray? = withContext(Dispatchers.IO) {
        try {
            if (isRecording) {
                isRecording = false
                
                audioRecord?.let { record ->
                    if (record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                        record.stop()
                    }
                }
                
                val audioData = audioBuffer.toByteArray()
                Log.d(TAG, "Stopped recording, captured ${audioData.size} bytes")
                
                return@withContext if (audioData.isNotEmpty()) {
                    // Convert PCM16 to float32 and normalize to [-1, 1] range as required by MediaPipe
                    val processedAudio = convertPcm16ToFloat32AndNormalize(audioData)
                    convertToWav(processedAudio)
                } else {
                    null
                }
            }
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recording", e)
            null
        }
    }
    
    /**
     * Convert PCM16 audio data to float32 format and normalize to [-1, 1] range
     * as required by MediaPipe Gemma-3n models
     */
    private fun convertPcm16ToFloat32AndNormalize(pcm16Data: ByteArray): ByteArray {
        val samples = pcm16Data.size / 2 // Each sample is 2 bytes (16-bit)
        val float32Data = ByteArray(samples * 4) // Each float32 is 4 bytes
        val buffer = ByteBuffer.wrap(float32Data).order(ByteOrder.LITTLE_ENDIAN)
        
        for (i in 0 until samples) {
            // Read 16-bit sample (little-endian)
            val sample16 = (pcm16Data[i * 2].toInt() and 0xFF) or 
                          (pcm16Data[i * 2 + 1].toInt() shl 8)
            
            // Convert from signed 16-bit to signed short
            val signedSample = sample16.toShort()
            
            // Normalize to [-1, 1] range by dividing by 32768 (2^15)
            val normalizedSample = signedSample.toFloat() / 32768.0f
            
            // Clamp to [-1, 1] range to handle edge cases
            val clampedSample = normalizedSample.coerceIn(-1.0f, 1.0f)
            
            // Write float32 value
            buffer.putFloat(clampedSample)
        }
        
        Log.d(TAG, "Converted ${pcm16Data.size} bytes PCM16 to ${float32Data.size} bytes float32, ${samples} samples")
        return float32Data
    }

    override suspend fun convertToWav(audioData: ByteArray): ByteArray = withContext(Dispatchers.IO) {
        try {
            // For MediaPipe, we need WAV format with float32 data
            // audioData is now already float32 normalized data
            val wavHeader = createFloat32WavHeader(audioData.size)
            
            // Combine header and audio data
            val wavData = ByteArray(wavHeader.size + audioData.size)
            System.arraycopy(wavHeader, 0, wavData, 0, wavHeader.size)
            System.arraycopy(audioData, 0, wavData, wavHeader.size, audioData.size)
            
            Log.d(TAG, "Converted ${audioData.size} bytes to float32 WAV format (${wavData.size} bytes total)")
            wavData
        } catch (e: Exception) {
            Log.e(TAG, "Error converting to WAV", e)
            audioData // Return raw data if conversion fails
        }
    }
    
    private fun createWavHeader(audioDataSize: Int): ByteArray {
        val header = ByteArray(44)
        val buffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
        
        // RIFF header
        buffer.put("RIFF".toByteArray()) // ChunkID
        buffer.putInt(36 + audioDataSize) // ChunkSize
        buffer.put("WAVE".toByteArray()) // Format
        
        // fmt subchunk
        buffer.put("fmt ".toByteArray()) // Subchunk1ID
        buffer.putInt(16) // Subchunk1Size (16 for PCM)
        buffer.putShort(1) // AudioFormat (1 for PCM)
        buffer.putShort(1) // NumChannels (1 for mono)
        buffer.putInt(SAMPLE_RATE) // SampleRate
        buffer.putInt(SAMPLE_RATE * 1 * 16 / 8) // ByteRate
        buffer.putShort((1 * 16 / 8).toShort()) // BlockAlign
        buffer.putShort(16) // BitsPerSample
        
        // data subchunk
        buffer.put("data".toByteArray()) // Subchunk2ID
        buffer.putInt(audioDataSize) // Subchunk2Size
        
        return header
    }
    
    private fun createFloat32WavHeader(audioDataSize: Int): ByteArray {
        val header = ByteArray(44)
        val buffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
        
        // RIFF header
        buffer.put("RIFF".toByteArray()) // ChunkID
        buffer.putInt(36 + audioDataSize) // ChunkSize
        buffer.put("WAVE".toByteArray()) // Format
        
        // fmt subchunk
        buffer.put("fmt ".toByteArray()) // Subchunk1ID
        buffer.putInt(16) // Subchunk1Size (PCM)
        buffer.putShort(3) // AudioFormat (3 = IEEE float)
        buffer.putShort(1) // NumChannels (mono)
        buffer.putInt(SAMPLE_RATE) // SampleRate
        buffer.putInt(SAMPLE_RATE * 1 * 32 / 8) // ByteRate (32-bit float)
        buffer.putShort((1 * 32 / 8).toShort()) // BlockAlign
        buffer.putShort(32) // BitsPerSample (32-bit float)
        
        // data subchunk
        buffer.put("data".toByteArray()) // Subchunk2ID
        buffer.putInt(audioDataSize) // Subchunk2Size
        
        return header
    }
    
    private fun calculateAudioLevel(buffer: ByteArray, bytesRead: Int): Float {
        var sum = 0L
        var samples = 0
        
        // Convert bytes to 16-bit samples and calculate RMS
        for (i in 0 until bytesRead - 1 step 2) {
            val sample = (buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8)
            sum += (sample * sample).toLong()
            samples++
        }
        
        if (samples == 0) return 0f
        
        val rms = kotlin.math.sqrt(sum.toDouble() / samples)
        
        // Normalize to 0-1 range (adjust based on typical microphone levels)
        return (rms / 32767.0).toFloat().coerceIn(0f, 1f)
    }
    
    override fun cleanup() {
        try {
            isRecording = false
            
            audioRecord?.let { record ->
                if (record.state == AudioRecord.STATE_INITIALIZED) {
                    if (record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                        record.stop()
                    }
                    record.release()
                }
            }
            audioRecord = null
            
            audioBuffer.reset()
            
            Log.d(TAG, "Audio recording cleanup completed")
        } catch (e: Exception) {
            Log.e(TAG, "Error during audio cleanup", e)
        }
    }
}
