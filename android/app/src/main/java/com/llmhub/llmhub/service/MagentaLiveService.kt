package com.llmhub.llmhub.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaMetadata
import android.media.session.MediaSession
import android.media.session.PlaybackState
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import com.llmhub.llmhub.MainActivity
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.MagentaRealtime
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Live Magenta RealTime 2 generation that keeps playing with the screen locked, like a music app: a mediaPlayback
 * foreground service with a MediaSession, so the system shows its media card (play/pause/stop) on the lock screen
 * and in the notification shade. 0.4 s chunks stream into an AudioTrack whose blocking writes keep generation
 * about [BUFFER_SECONDS] ahead of playback. The prompt can change while it plays; the piece bends toward it.
 */
class MagentaLiveService : Service() {

    data class LiveState(
        val active: Boolean = false,
        val starting: Boolean = false,
        val paused: Boolean = false,
        val playedSeconds: Int = 0,
        val error: String? = null,
        /** The finished session's audio (16-bit 48 kHz stereo WAV) and the prompt it started with. */
        val recording: java.io.File? = null,
        val recordingPrompt: String = ""
    )

    companion object {
        private const val TAG = "MagentaLiveService"
        private const val CHANNEL_ID = "music_live_generation"
        private const val NOTIFICATION_ID = 7421
        private const val BUFFER_SECONDS = 1.5
        private const val ACTION_START = "com.llmhub.llmhub.magenta.START"
        private const val ACTION_PAUSE = "com.llmhub.llmhub.magenta.PAUSE"
        private const val ACTION_RESUME = "com.llmhub.llmhub.magenta.RESUME"
        private const val ACTION_STOP = "com.llmhub.llmhub.magenta.STOP"
        private const val EXTRA_MAX_SECONDS = "max_seconds"

        private val _state = MutableStateFlow(LiveState())
        val state: StateFlow<LiveState> = _state.asStateFlow()

        /** Read by the generation loop before every request, so a new prompt steers the next 0.4 s. */
        @Volatile var prompt: String = ""
            private set

        /** [maxSeconds] null = until stopped. */
        fun start(context: Context, prompt: String, maxSeconds: Int?) {
            this.prompt = prompt
            _state.value = LiveState(active = true, starting = true)  // a new session; the last recording was taken
            val intent = Intent(context, MagentaLiveService::class.java).setAction(ACTION_START)
                .putExtra(EXTRA_MAX_SECONDS, maxSeconds ?: -1)
            context.startForegroundService(intent)
        }

        fun updatePrompt(prompt: String) {
            if (prompt.isNotBlank()) this.prompt = prompt
        }

        fun pause(context: Context) = send(context, ACTION_PAUSE)
        fun resume(context: Context) = send(context, ACTION_RESUME)
        fun stop(context: Context) {
            if (_state.value.active) send(context, ACTION_STOP)
        }

        private fun send(context: Context, action: String) {
            context.startService(Intent(context, MagentaLiveService::class.java).setAction(action))
        }
    }

    @Volatile private var running = false
    @Volatile private var paused = false
    private var worker: Thread? = null
    private var track: AudioTrack? = null
    private var session: MediaSession? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var focusRequest: AudioFocusRequest? = null
    private var pausedByFocusLoss = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startLive(intent.getIntExtra(EXTRA_MAX_SECONDS, -1).takeIf { it > 0 })
            ACTION_PAUSE -> setPaused(true)
            ACTION_RESUME -> setPaused(false)
            ACTION_STOP -> stopLive()
        }
        return START_NOT_STICKY
    }

    private fun startLive(maxSeconds: Int?) {
        createChannel()
        val mediaSession = MediaSession(this, "MagentaLive").apply {
            setCallback(object : MediaSession.Callback() {
                override fun onPlay() = setPaused(false)
                override fun onPause() = setPaused(true)
                override fun onStop() = stopLive()
            })
            isActive = true
        }
        session = mediaSession
        updateSession()
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        if (running) return
        running = true
        paused = false
        wakeLock = (getSystemService(POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "LLMHub:MagentaLive").apply { acquire() }
        requestFocus()
        worker = Thread({ generate(maxSeconds) }, "magenta-live").apply { start() }
    }

    private fun generate(maxSeconds: Int?) {
        var audio: AudioTrack? = null
        val startPrompt = prompt
        val recording = java.io.File(java.io.File(cacheDir, "magenta_rt2").apply { mkdirs() }, "live_${System.currentTimeMillis()}.wav")
        val recorder = java.io.RandomAccessFile(recording, "rw").apply { setLength(0); write(ByteArray(44)) }
        var recordedBytes = 0L
        try {
            val engine = MagentaRealtime.acquire(this, "live")
            val bytesPerSecond = MagentaRealtime.SAMPLE_RATE * MagentaRealtime.CHANNELS * 2
            audio = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(MagentaRealtime.SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                        .build()
                )
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes((bytesPerSecond * BUFFER_SECONDS).toInt())
                .build()
            track = audio
            var generatedBytes = 0L
            var first = true
            var lastReport = 0L
            while (running) {
                if (paused) {
                    SystemClock.sleep(50)
                    continue
                }
                if (maxSeconds != null && generatedBytes >= maxSeconds.toLong() * bytesPerSecond) break
                val wav = engine.request(MagentaRealtime.liveRequest(prompt, newPiece = first))
                val pcm = try { MagentaRealtime.readPcm(wav) } finally { wav.delete() }
                first = false
                recorder.write(pcm)
                recordedBytes += pcm.size
                var offset = 0
                while (offset < pcm.size && running) {
                    if (paused) { SystemClock.sleep(50); continue }
                    val written = audio.write(pcm, offset, pcm.size - offset, AudioTrack.WRITE_BLOCKING)
                    if (written < 0) error("AudioTrack write failed: $written")
                    offset += written
                    if (audio.playState != AudioTrack.PLAYSTATE_PLAYING && !paused) audio.play()
                }
                generatedBytes += pcm.size
                val now = SystemClock.elapsedRealtime()
                if (now - lastReport > 500) {
                    lastReport = now
                    publish(audio)
                }
            }
            // A finite session: let what is already queued play out. In MODE_STREAM the last partial buffer only
            // plays once stop() is called (stop() drains, then stops), so stop first and wait for the head.
            // The head position resets once stopped, so wait for what was still queued at that moment.
            if (running && maxSeconds != null) {
                val totalFrames = generatedBytes / (MagentaRealtime.CHANNELS * 2)
                val queuedFrames = totalFrames - (audio.playbackHeadPosition.toLong() and 0xffffffffL)
                audio.stop()
                val drainEnd = SystemClock.elapsedRealtime() + queuedFrames * 1000 / MagentaRealtime.SAMPLE_RATE + 300
                while (running && SystemClock.elapsedRealtime() < drainEnd) SystemClock.sleep(100)
                _state.value = _state.value.copy(playedSeconds = (totalFrames / MagentaRealtime.SAMPLE_RATE).toInt())
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Live generation failed", t)
            if (running) _state.value = _state.value.copy(error = t.message ?: t.javaClass.simpleName)
        } finally {
            runCatching { audio?.stop() }
            runCatching { audio?.release() }
            track = null
            val saved = runCatching {
                recorder.seek(0)
                recorder.write(wavHeader(recordedBytes))
                recorder.close()
                recordedBytes > 0
            }.getOrDefault(false)
            if (saved) _state.value = _state.value.copy(recording = recording, recordingPrompt = startPrompt)
            else { runCatching { recorder.close() }; recording.delete() }
            MagentaRealtime.release("live")
            android.os.Handler(mainLooper).post { finishService() }
        }
    }

    private val timestamp = android.media.AudioTimestamp()
    private var shownPrompt = ""

    private fun wavHeader(dataBytes: Long): ByteArray {
        val rate = MagentaRealtime.SAMPLE_RATE
        val channels = MagentaRealtime.CHANNELS
        return java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII)); putInt((36 + dataBytes).toInt())
            put("WAVE".toByteArray(Charsets.US_ASCII)); put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16); putShort(1); putShort(channels.toShort()); putInt(rate)
            putInt(rate * channels * 2); putShort((channels * 2).toShort()); putShort(16)
            put("data".toByteArray(Charsets.US_ASCII)); putInt(dataBytes.toInt())
        }.array()
    }

    private fun publish(audio: AudioTrack) {
        // getTimestamp() counts frames actually presented. playbackHeadPosition counts frames the mixer has taken,
        // which runs seconds ahead with the screen off (Android then drains audio into a larger low-power buffer).
        val hasTimestamp = audio.getTimestamp(timestamp)
        val head = audio.playbackHeadPosition.toLong() and 0xffffffffL
        val frames = if (hasTimestamp) timestamp.framePosition else head
        val now = SystemClock.elapsedRealtime()
        if (now - lastPositionLog >= 5000) {
            lastPositionLog = now
            Log.d(TAG, "position timestamp=${if (hasTimestamp) timestamp.framePosition else -1} head=$head underruns=${audio.underrunCount}")
        }
        val played = maxOf(frames / MagentaRealtime.SAMPLE_RATE, _state.value.playedSeconds.toLong())
        _state.value = _state.value.copy(starting = false, playedSeconds = played.toInt(), paused = paused)
        if (prompt != shownPrompt) {
            // The lock-screen card shows the prompt: refresh it when the piece is steered somewhere else.
            shownPrompt = prompt
            updateSession(played * 1000, metadata = true)
            getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification())
        } else {
            updateSession(played * 1000, metadata = false)
        }
        if (audio.underrunCount > reportedUnderruns) {
            reportedUnderruns = audio.underrunCount
            Log.w(TAG, "AudioTrack underruns: $reportedUnderruns at ${played}s")
        }
    }
    private var reportedUnderruns = 0
    private var lastPositionLog = 0L

    private fun setPaused(value: Boolean) {
        if (!running) return
        paused = value
        pausedByFocusLoss = false
        if (value) track?.pause() else track?.play()
        _state.value = _state.value.copy(paused = value)
        updateSession()
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification())
    }

    /** The loop sees this before its next 0.4 s request (no interrupt: that would desync a request in flight). */
    private fun stopLive() {
        running = false
        if (worker == null) finishService()
    }

    private fun finishService() {
        running = false
        worker = null
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
        focusRequest?.let { getSystemService(AudioManager::class.java).abandonAudioFocusRequest(it) }
        focusRequest = null
        session?.run { isActive = false; release() }
        session = null
        val ended = _state.value
        _state.value = LiveState(error = ended.error, recording = ended.recording, recordingPrompt = ended.recordingPrompt)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        // Leaving the app ends live generation, as every generating feature does.
        stopLive()
        super.onTaskRemoved(rootIntent)
    }

    override fun onDestroy() {
        running = false
        super.onDestroy()
    }

    private fun requestFocus() {
        val request = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            .setOnAudioFocusChangeListener { change ->
                when (change) {
                    AudioManager.AUDIOFOCUS_LOSS -> setPaused(true)
                    AudioManager.AUDIOFOCUS_LOSS_TRANSIENT -> if (!paused) {
                        setPaused(true)
                        pausedByFocusLoss = true
                    }
                    AudioManager.AUDIOFOCUS_GAIN -> if (pausedByFocusLoss) setPaused(false)
                }
            }
            .build()
        focusRequest = request
        getSystemService(AudioManager::class.java).requestAudioFocus(request)
    }

    private fun updateSession(positionMs: Long = (_state.value.playedSeconds * 1000L), metadata: Boolean = true) {
        val mediaSession = session ?: return
        if (metadata) mediaSession.setMetadata(
            MediaMetadata.Builder()
                .putString(MediaMetadata.METADATA_KEY_TITLE, prompt)
                .putString(MediaMetadata.METADATA_KEY_ARTIST, MagentaRealtime.MODEL_NAME)
                .putString(MediaMetadata.METADATA_KEY_ALBUM, getString(R.string.music_live_generation))
                .putLong(MediaMetadata.METADATA_KEY_DURATION, -1L)
                .build()
        )
        // (setPlaybackState below runs every time.)
        mediaSession.setPlaybackState(
            PlaybackState.Builder()
                .setActions(
                    PlaybackState.ACTION_PLAY or PlaybackState.ACTION_PAUSE or
                        PlaybackState.ACTION_PLAY_PAUSE or PlaybackState.ACTION_STOP
                )
                .setState(
                    if (paused) PlaybackState.STATE_PAUSED else PlaybackState.STATE_PLAYING,
                    positionMs,
                    if (paused) 0f else 1f
                )
                .build()
        )
    }

    private fun buildNotification(): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        fun action(action: String, code: Int) = PendingIntent.getService(
            this, code, Intent(this, MagentaLiveService::class.java).setAction(action), PendingIntent.FLAG_IMMUTABLE
        )
        val toggle = if (paused) {
            Notification.Action.Builder(
                android.graphics.drawable.Icon.createWithResource(this, android.R.drawable.ic_media_play),
                getString(R.string.music_live_resume), action(ACTION_RESUME, 1)
            ).build()
        } else {
            Notification.Action.Builder(
                android.graphics.drawable.Icon.createWithResource(this, android.R.drawable.ic_media_pause),
                getString(R.string.music_live_pause), action(ACTION_PAUSE, 2)
            ).build()
        }
        val stop = Notification.Action.Builder(
            android.graphics.drawable.Icon.createWithResource(this, android.R.drawable.ic_menu_close_clear_cancel),
            getString(R.string.music_stop_generation), action(ACTION_STOP, 3)
        ).build()
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(prompt)
            .setContentText(getString(R.string.music_live_generation))
            .setContentIntent(open)
            .setOngoing(true)
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .setCategory(Notification.CATEGORY_TRANSPORT)
            .addAction(toggle)
            .addAction(stop)
            .setStyle(Notification.MediaStyle().setMediaSession(session?.sessionToken).setShowActionsInCompactView(0, 1))
            .build()
    }

    private fun createChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL_ID) == null) {
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, getString(R.string.music_live_generation), NotificationManager.IMPORTANCE_LOW)
            )
        }
    }
}
