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
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.ByteArrayOutputStream

class AudioInputService(private val context: Context) {
    private var audioRecord: AudioRecord? = null
    private var outputFile: File? = null        // temp wav for preview / debugging
    private val pcmStream = ByteArrayOutputStream()
    private var isRecording = false
    private var recordingThread: Thread? = null
    private var recordingStartTime: Long = 0
    @Volatile
    private var captureHadSpeech: Boolean = false
    var maxRecordingReached: Boolean = false
        private set
    var maxDurationMs: Long = 29500L
    
    // Coroutine scope for callbacks
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    
    // Expose elapsed time as a flow for UI updates
    private val _elapsedTimeMs = MutableStateFlow(0L)
    val elapsedTimeMs: StateFlow<Long> = _elapsedTimeMs.asStateFlow()

    // Expose normalized live mic level (0f..1f) for waveform/reactive UI.
    private val _audioLevel = MutableStateFlow(0f)
    val audioLevel: StateFlow<Float> = _audioLevel.asStateFlow()
    
    // Callback for when recording stops automatically
    var onRecordingAutoStopped: (() -> Unit)? = null

    // When false, silence detection will NOT auto-stop recording (timer expiry still stops it)
    var silenceAutoStopEnabled: Boolean = true
    
    companion object {
        private const val TAG = "AudioInputService"
        private const val SAMPLE_RATE = 16000 // 16kHz for speech
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO // Mono channel
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT // 16-bit PCM
        private val BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        private const val MAX_RECORDING_DURATION_MS = 29500L // 29.5 seconds
        private const val SPEECH_START_THRESHOLD = 0.075f
        private const val SILENCE_THRESHOLD = 0.030f
        private const val SILENCE_STOP_DURATION_MS = 1150L
        private const val MIN_ACTIVE_SPEECH_MS = 450L
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
            outputFile = File.createTempFile("audio_", ".wav", context.cacheDir) // optional WAV copy

            // Reset PCM buffer
            synchronized(pcmStream) {
                pcmStream.reset()
            }
            
            // Initialize AudioRecord for mono PCM recording (MediaPipe expects raw 16-bit PCM)
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
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
            
            // Reset max recording flag and record start time
            maxRecordingReached = false
            captureHadSpeech = false
            recordingStartTime = System.currentTimeMillis()
            _audioLevel.value = 0f
            
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
    suspend fun stopRecording(forceKeep: Boolean = false): ByteArray? = withContext(Dispatchers.IO) {
        try {
            // Allow stopping even if isRecording is false (for auto-stop case)
            // Check if we have an active recording thread or audio data
            val pcmData = synchronized(pcmStream) {
                if (audioRecord == null && pcmStream.size() == 0) {
                    Log.w(TAG, "No recording in progress or already stopped")
                    return@withContext null
                }
                pcmStream.toByteArray()
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
            
            Log.d(TAG, "Stopped recording, PCM16 size: ${pcmData.size} bytes")

            if (!captureHadSpeech && !forceKeep) {
                Log.d(TAG, "Discarding capture because no speech was detected")
                synchronized(pcmStream) {
                    pcmStream.reset()
                }
                _audioLevel.value = 0f
                outputFile?.delete()
                outputFile = null
                return@withContext null
            }

            // Convert PCM16 to float32 WAV (MediaPipe requirement)
            val float32Wav = convertPcm16ToFloat32Wav(pcmData)

            // Clear PCM buffer for next recording
            synchronized(pcmStream) {
                pcmStream.reset()
            }
            _audioLevel.value = 0f

            // Cleanup temp WAV file
            outputFile?.delete()
            outputFile = null
            
            return@withContext float32Wav
            
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

    /**
     * Get the float32 WAV data recorded so far (thread-safe copy).
     */
    fun getFloat32WavDataSoFar(): ByteArray? {
        val pcmData = synchronized(pcmStream) {
            if (pcmStream.size() == 0) return null
            pcmStream.toByteArray()
        }
        return convertPcm16ToFloat32Wav(pcmData)
    }
    
    private fun cleanup() {
        try {
            isRecording = false
            maxRecordingReached = false
            recordingStartTime = 0
            captureHadSpeech = false
            _elapsedTimeMs.value = 0L
            _audioLevel.value = 0f
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
        synchronized(pcmStream) {
            pcmStream.reset() // Ensure PCM buffer is empty on cleanup
        }
    }
    
    /**
     * Write audio data to WAV file
     */
    private fun writeAudioDataToFile() {
        val data = ByteArray(BUFFER_SIZE)
        var output: FileOutputStream? = null
        val outputTarget = outputFile
        if (outputTarget == null) {
            Log.e(TAG, "Recording output file is null before capture starts")
            return
        }
        var hasSpeechStarted = false
        var firstSpeechAtMs = 0L
        var lastSpeechAtMs = 0L
        
        try {
            output = FileOutputStream(outputTarget)
            
            // Write WAV header (we'll update it later with actual data size)
            writeWavHeader(output, SAMPLE_RATE, 1, 16, 0) // 0 data size initially
            
            var totalDataSize = 0
            
            while (isRecording) {
                // Check if max duration reached and update elapsed time
                val elapsedTime = System.currentTimeMillis() - recordingStartTime
                _elapsedTimeMs.value = elapsedTime
                
                if (elapsedTime >= maxDurationMs) {
                    Log.d(TAG, "Max recording duration reached ($maxDurationMs ms)")
                    maxRecordingReached = true
                    isRecording = false
                    // Notify UI that recording stopped automatically (on main thread)
                    serviceScope.launch {
                        onRecordingAutoStopped?.invoke()
                    }
                    break
                }
                
                val bytesRead = audioRecord?.read(data, 0, BUFFER_SIZE) ?: 0
                if (bytesRead > 0) {
                    val level = calculateNormalizedAudioLevel(data, bytesRead)
                    _audioLevel.value = level

                    val now = System.currentTimeMillis()
                    if (!hasSpeechStarted) {
                        if (level >= SPEECH_START_THRESHOLD) {
                            hasSpeechStarted = true
                            captureHadSpeech = true
                            firstSpeechAtMs = now
                            lastSpeechAtMs = now
                        }
                    } else {
                        if (level >= SILENCE_THRESHOLD) {
                            lastSpeechAtMs = now
                        } else {
                            val speechAgeMs = now - firstSpeechAtMs
                            val silenceMs = now - lastSpeechAtMs
                            if (silenceAutoStopEnabled && speechAgeMs >= MIN_ACTIVE_SPEECH_MS && silenceMs >= SILENCE_STOP_DURATION_MS) {
                                Log.d(TAG, "Detected end of speech (silence ${silenceMs}ms), auto-stopping recording")
                                isRecording = false
                                serviceScope.launch {
                                    onRecordingAutoStopped?.invoke()
                                }
                                break
                            }
                        }
                    }

                    // Write raw PCM to in-memory buffer for MediaPipe
                    synchronized(pcmStream) {
                        pcmStream.write(data, 0, bytesRead)
                    }

                    // Also write to WAV file on disk (optional, retains header)
                    output.write(data, 0, bytesRead)
                    totalDataSize += bytesRead
                }
            }
            
            output.close()
            
            // Update WAV header with actual data size
            updateWavHeader(outputTarget, totalDataSize)
            
        } catch (e: IOException) {
            Log.e(TAG, "Error writing audio data", e)
        } finally {
            _audioLevel.value = 0f
            output?.close()
        }
    }

    private fun calculateNormalizedAudioLevel(buffer: ByteArray, bytesRead: Int): Float {
        if (bytesRead < 2) return 0f

        var sumSquares = 0.0
        var sampleCount = 0
        var i = 0
        while (i + 1 < bytesRead) {
            val sample = ((buffer[i + 1].toInt() shl 8) or (buffer[i].toInt() and 0xFF)).toShort().toInt()
            val normalized = sample / 32768.0
            sumSquares += normalized * normalized
            sampleCount++
            i += 2
        }

        if (sampleCount == 0) return 0f
        val rms = kotlin.math.sqrt(sumSquares / sampleCount)

        // Lightly boost conversational speech to make UI response visible.
        return (rms * 4.8).toFloat().coerceIn(0f, 1f)
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

    // Helper to create 32-bit float mono WAV header (little-endian) at 16 kHz
    private fun createFloat32WavHeader(audioDataSize: Int): ByteArray {
        val header = ByteArray(44)
        val buffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF chunk descriptor
        buffer.put("RIFF".toByteArray())
        buffer.putInt(36 + audioDataSize) // ChunkSize = 4 + (8 + SubChunk1) + (8 + SubChunk2)
        buffer.put("WAVE".toByteArray())

        // fmt sub-chunk
        buffer.put("fmt ".toByteArray())
        buffer.putInt(16)               // SubChunk1Size for PCM
        buffer.putShort(3)              // AudioFormat 3 = IEEE float
        buffer.putShort(1)              // NumChannels (mono)
        buffer.putInt(SAMPLE_RATE)      // SampleRate
        val byteRate = SAMPLE_RATE * 1 * 32 / 8
        buffer.putInt(byteRate)         // ByteRate
        buffer.putShort((1 * 32 / 8).toShort()) // BlockAlign
        buffer.putShort(32)             // BitsPerSample

        // data sub-chunk
        buffer.put("data".toByteArray())
        buffer.putInt(audioDataSize)

        return header
    }

    // Convert signed PCM16 little-endian mono to float32 WAV mono 16 kHz
    private fun convertPcm16ToFloat32Wav(pcm16: ByteArray): ByteArray {
        val samples = pcm16.size / 2
        val floatBytes = ByteArray(samples * 4)
        val floatBuf = ByteBuffer.wrap(floatBytes).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until samples) {
            val s = (pcm16[i*2].toInt() and 0xFF) or (pcm16[i*2+1].toInt() shl 8)
            val signed = s.toShort()
            val normalized = signed.toFloat() / 32768.0f
            floatBuf.putFloat(normalized)
        }
        val header = createFloat32WavHeader(floatBytes.size)
        val out = ByteArray(header.size + floatBytes.size)
        System.arraycopy(header,0,out,0,header.size)
        System.arraycopy(floatBytes,0,out,header.size,floatBytes.size)
        return out
    }
}
