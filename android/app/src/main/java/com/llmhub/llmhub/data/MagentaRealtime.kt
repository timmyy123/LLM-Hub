package com.llmhub.llmhub.data

import android.content.Context
import android.system.Os
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * Magenta RealTime 2 (Google DeepMind, `mrt2_small`, 230M) on the Adreno GPU through the adreno-llms OpenCL engine
 * (Apache 2.0, built by tools/magenta_rt2/build.sh). The engine runs as its own process in `--serve` mode, the way
 * adreno-llms' own app drives it: weights and kernel JIT are paid once, then each stdin request generates the next
 * stretch of the same piece and writes it to a WAV in the working folder. A GPU fault ends that process, not the app.
 *
 * It is offered on the Snapdragons where GGUF models use the GPU (DeviceInfo.isGgufGpuSupported). Live generation needs it faster than playback, which only [LIVE_SOCS] are
 * (8 Elite Gen 5 1.31-1.39x real time, 8 Elite Extreme Gen 6 1.42-1.65x); elsewhere a clip is generated first and
 * then played (30 s took 32 s on the 8 Elite, 56 s on the 8 Gen 2, ~2 min on the 8 Gen 3).
 */
object MagentaRealtime {
    private const val TAG = "MagentaRealtime"

    const val MODEL_NAME = "Magenta RealTime 2 Small"
    const val SAMPLE_RATE = 48_000
    const val CHANNELS = 2
    const val FRAMES_PER_SECOND = 25
    /** Live generation: 8 Elite Gen 5 and the 8 Elite Gen 6 family (Adreno 840/850). */
    val LIVE_SOCS = setOf("SM8850", "SM8850P", "SM8950", "SM8975")

    fun isLiveSupported(): Boolean = ModelData.socForMatching() in LIVE_SOCS
    /** adreno-llms' benchmarked live settings (0.4 s per request). */
    const val LIVE_FRAMES = 10
    private const val LIVE_SETTINGS = "chunk=10 pipeline=0 temperature=1.1 topk=50 cfgtok=5,0,4 mctail=6"
    private const val EXECUTABLE = "libmagenta_rt2.so"

    fun isModel(modelName: String): Boolean = modelName.startsWith("Magenta RealTime", ignoreCase = true)

    fun modelDir(context: Context): File =
        File(File(context.filesDir, "models"), MODEL_NAME.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), ""))

    private val holders = mutableSetOf<String>()
    private var engine: Engine? = null

    /** Starts the engine if needed (loads ~0.8 GB of weights and JITs the kernels) and registers [holder]. */
    @Synchronized
    fun acquire(context: Context, holder: String): Engine {
        val running = engine?.takeIf { it.isAlive }
        val current = running ?: Engine.start(context.applicationContext).also { engine = it }
        holders += holder
        return current
    }

    /** Drops [holder]; the process quits once nothing holds it. */
    @Synchronized
    fun release(holder: String) {
        holders -= holder
        if (holders.isEmpty()) {
            engine?.close()
            engine = null
        }
    }

    /** [seed] starts a new piece from a fixed seed (the Seed setting); null keeps the engine's own. */
    fun liveRequest(prompt: String, newPiece: Boolean, seed: Long? = null): String =
        "frames=$LIVE_FRAMES $LIVE_SETTINGS${if (newPiece) " reset=1" + seedArg(seed) else ""} prompt=${clean(prompt)}"

    fun renderRequest(prompt: String, seconds: Int, seed: Long? = null): String =
        "frames=${seconds * FRAMES_PER_SECOND} chunk=10 pipeline=0 reset=1${seedArg(seed)} prompt=${clean(prompt)}"

    private fun seedArg(seed: Long?) = if (seed != null) " seed=${seed and 0xffffffffL}" else ""

    /** The prompt takes the rest of the request line, so it must stay on one line. */
    private fun clean(prompt: String) = prompt.replace(Regex("\\s+"), " ").trim()

    class Engine private constructor(private val process: Process, private val workDir: File) {
        private val events = LinkedBlockingQueue<String>()
        private val writer = process.outputStream.bufferedWriter()
        val isAlive: Boolean get() = process.isAlive

        init {
            Thread({
                try {
                    process.inputStream.bufferedReader().useLines { lines ->
                        lines.forEach { line ->
                            when {
                                line.startsWith("SERVE_") -> { Log.i(TAG, line); events.put(line) }
                                line.startsWith("MRT_PROGRESS ") -> events.put(line)
                                line.startsWith("BENCH ") || line.startsWith("DEVICE ") || line.contains("ERROR") -> Log.i(TAG, line)
                            }
                        }
                    }
                } catch (_: Exception) {
                } finally {
                    events.put("EXIT")
                }
            }, "magenta-rt2-output").apply { isDaemon = true }.start()
        }

        /**
         * Sends one request and returns the WAV it wrote; the caller deletes it. [onProgress] gets 0..1 from the
         * engine's MRT_PROGRESS lines (token frames, then the codec).
         */
        @Synchronized
        fun request(line: String, timeoutSeconds: Long = 600, onProgress: ((Float) -> Unit)? = null): File {
            check(process.isAlive) { "Magenta engine stopped" }
            events.clear()
            writer.write(line)
            writer.newLine()
            writer.flush()
            while (true) {
                val event = events.poll(timeoutSeconds, TimeUnit.SECONDS) ?: error("Magenta engine timed out")
                when {
                    event.startsWith("SERVE_DONE") -> {
                        val name = Regex("wav=(\\S+)").find(event)?.groupValues?.get(1) ?: error("No WAV in $event")
                        return File(workDir, name)
                    }
                    event.startsWith("SERVE_ERROR") -> error("Magenta engine: $event")
                    event.startsWith("MRT_PROGRESS ") && onProgress != null -> {
                        val m = Regex("stage=(\\w+) frame=(\\d+)/(\\d+)").find(event) ?: continue
                        val done = m.groupValues[2].toFloat() / m.groupValues[3].toFloat().coerceAtLeast(1f)
                        onProgress((if (m.groupValues[1] == "codec") 0.85f + 0.15f * done else 0.85f * done).coerceIn(0f, 1f))
                    }
                    event == "EXIT" -> error("Magenta engine exited")
                }
            }
        }

        fun close() {
            try {
                writer.write("quit")
                writer.newLine()
                writer.flush()
            } catch (_: Exception) {
            }
            if (!process.waitFor(3, TimeUnit.SECONDS)) process.destroy()
            workDir.listFiles { f -> f.name.endsWith(".wav") }?.forEach { it.delete() }
        }

        private fun awaitReady(timeoutSeconds: Long): Boolean {
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(timeoutSeconds)
            while (true) {
                val left = deadline - System.nanoTime()
                if (left <= 0) return false
                val event = events.poll(left, TimeUnit.NANOSECONDS) ?: return false
                if (event == "SERVE_READY") return true
                if (event == "EXIT") return false
            }
        }

        companion object {
            fun start(context: Context): Engine {
                val modelDir = modelDir(context)
                require(File(modelDir, "model.fp16.bin").isFile) { "Magenta RealTime 2 is not downloaded" }
                val nativeDir = context.applicationInfo.nativeLibraryDir
                val executable = File(nativeDir, EXECUTABLE)
                require(executable.isFile) { "Missing $EXECUTABLE" }
                val workDir = prepareWorkDir(context, modelDir)
                val process = ProcessBuilder(executable.absolutePath, "--serve")
                    .directory(workDir)
                    .redirectErrorStream(true)
                    .apply {
                        environment()["LD_LIBRARY_PATH"] = "$nativeDir:/vendor/lib64:/system/vendor/lib64"
                        // fp16 codec everywhere: the engine only defaults to it on Adreno 8xx, and on the 8 Gen 2 it cut
                        // a 10 s render from 20.6 s to 17.7 s (its built-in fp16-vs-fp32 codec check passes there).
                        environment()["NNOPT_CODECFP16"] = "1"
                    }
                    .start()
                val engine = Engine(process, workDir)
                val started = System.nanoTime()
                if (!engine.awaitReady(180)) {
                    process.destroy()
                    error("Magenta engine did not start")
                }
                Log.i(TAG, "Engine ready in ${(System.nanoTime() - started) / 1_000_000} ms")
                return engine
            }

            /** The engine reads kernels/ and weights/ relative to its working folder. */
            private fun prepareWorkDir(context: Context, modelDir: File): File {
                val work = File(context.filesDir, "magenta_rt2_run").apply { mkdirs() }
                val version = context.assets.open("magenta_rt2/VERSION").bufferedReader().readText().trim()
                val stamp = File(work, "VERSION")
                val kernels = File(work, "kernels")
                if (!stamp.isFile || stamp.readText().trim() != version) {
                    kernels.deleteRecursively()
                    kernels.mkdirs()
                    context.assets.list("magenta_rt2/kernels")?.forEach { name ->
                        context.assets.open("magenta_rt2/kernels/$name").use { input ->
                            File(kernels, name).outputStream().use { input.copyTo(it) }
                        }
                    }
                    stamp.writeText(version)
                }
                val weights = File(work, "weights")
                weights.delete() // the old link, never its target
                Os.symlink(modelDir.absolutePath, weights.absolutePath)
                work.listFiles { f -> f.name.endsWith(".wav") }?.forEach { it.delete() }
                return work
            }
        }
    }

    /** The interleaved 16-bit PCM of a WAV the engine wrote (48 kHz stereo). */
    fun readPcm(wav: File): ByteArray = RandomAccessFile(wav, "r").use { file ->
        var offset = 12L
        while (offset + 8 <= file.length()) {
            file.seek(offset)
            val id = ByteArray(4).also { file.readFully(it) }
            val size = Integer.reverseBytes(file.readInt()).toLong() and 0xffffffffL
            if (String(id, Charsets.US_ASCII) == "data") {
                val bytes = ByteArray(minOf(size, file.length() - offset - 8).toInt())
                file.readFully(bytes)
                return bytes
            }
            offset += 8 + size + (size and 1)
        }
        error("No data chunk in ${wav.name}")
    }
}
