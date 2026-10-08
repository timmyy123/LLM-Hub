#include "FFmpegCommand.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <malloc/malloc.h>
#include <mach/mach.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if __has_feature(address_sanitizer)
#include <sanitizer/allocator_interface.h>
#endif

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

typedef struct {
    atomic_bool cancelled;
    atomic_int polls;
} Cancellation;

static int cancelled(const void *opaque)
{
    Cancellation *value = (Cancellation *)opaque;
    atomic_fetch_add(&value->polls, 1);
    return atomic_load(&value->cancelled);
}

static char directory[PATH_MAX];
static const char *environment[] = {"AV_LOG_FORCE_NOCOLOR=1", NULL};

static FFmpegCommandContext context(void)
{
    FFmpegCommandContext c = {
        .input = tmpfile(), .output = tmpfile(), .error = tmpfile(),
        .directory = directory, .home = directory, .temporaryDirectory = directory,
        .environment = environment,
    };
    CHECK(c.input && c.output && c.error);
    return c;
}

static void close_context(FFmpegCommandContext *c)
{
    CHECK(!fclose(c->input)); CHECK(!fclose(c->output)); CHECK(!fclose(c->error));
}

static int run(FFmpegCommandContext *c, const char *const args[])
{
    int argc = 0;
    while (args[argc]) ++argc;
    char **copy = calloc(argc + 1, sizeof(*copy));
    CHECK(copy);
    for (int i = 0; i < argc; ++i) CHECK((copy[i] = strdup(args[i])));
    int status = !strcmp(args[0], "ffprobe") ? FFprobeCommandRun(argc, copy, c)
                                           : FFmpegCommandRun(argc, copy, c);
    for (int i = 0; i < argc; ++i) free(copy[i]);
    free(copy);
    return status;
}

static void expect(int status, const char *const args[])
{
    FFmpegCommandContext c = context();
    int actual = run(&c, args);
    if (status < 0 ? actual == 0 : actual != status) {
        rewind(c.error);
        char buffer[2048];
        size_t size;
        while ((size = fread(buffer, 1, sizeof(buffer), c.error)))
            fwrite(buffer, 1, size, stderr);
        fprintf(stderr, "Expected %d, got %d\n", status, actual);
        CHECK(0);
    }
    close_context(&c);
}

static unsigned descriptors(void)
{
    unsigned count = 0;
    for (int fd = 0; fd < 4096; ++fd) if (fcntl(fd, F_GETFD) >= 0) ++count;
    return count;
}

static unsigned threads(void)
{
    thread_act_array_t list;
    mach_msg_type_number_t count;
    CHECK(task_threads(mach_task_self(), &list, &count) == KERN_SUCCESS);
    for (unsigned i = 0; i < count; ++i) mach_port_deallocate(mach_task_self(), list[i]);
    vm_deallocate(mach_task_self(), (vm_address_t)list, count * sizeof(*list));
    return count;
}

static size_t allocated_bytes(void)
{
#if __has_feature(address_sanitizer)
    // malloc_zone_statistics also counts ASan's allocator/quarantine storage.
    return __sanitizer_get_current_allocated_bytes();
#else
    malloc_statistics_t statistics;
    malloc_zone_statistics(NULL, &statistics);
    return statistics.size_in_use;
#endif
}

static const char *audio[] = {"ffmpeg", "-v", "error", "-y", "-f", "lavfi",
    "-i", "sine=frequency=440:sample_rate=8000", "-t", "0.1",
    "-c:a", "pcm_s16le", "tone.wav", NULL};
static const char *version[] = {"ffmpeg", "-version", NULL};
static const char *probe_version[] = {"ffprobe", "-version", NULL};
static const char *probe_audio[] = {"ffprobe", "-v", "error", "-select_streams", "a:0",
    "-show_entries", "stream=codec_name,sample_rate:format=duration", "-of", "json",
    "/home/tone.wav", NULL};
static const char *probe_failure[] = {"ffprobe", "-v", "error", "-select_streams", "a:0",
    "-show_data_hash", "sha256", "-codec:a", "pcm_s16le", "-read_intervals", "%+1",
    "-show_entries", "stream=codec_name", "-of", "json=invalid_option=1",
    "-o", "probe-output.json", "tone.wav", NULL};
static const char *failure[] = {"ffmpeg", "-v", "error", "-progress", "progress.txt",
    "-i", "missing-file.wav", "unused.wav", NULL};

typedef struct {
    FFmpegCommandContext c;
    Cancellation cancellation;
    const char *const *args;
    atomic_bool finished;
    int status;
} Invocation;

static void *invoke(void *opaque)
{
    Invocation *i = opaque;
    i->status = run(&i->c, i->args);
    atomic_store(&i->finished, true);
    return NULL;
}

static void wait_flag(atomic_bool *flag)
{
    for (int i = 0; i < 500 && !atomic_load(flag); ++i) usleep(10000);
    CHECK(atomic_load(flag));
}

static void test_cancellation(void)
{
    // A writer stays open without producing data: the demuxer must be
    // interruptible even while waiting for the first byte of a media pipe.
    int pipefd[2]; CHECK(!pipe(pipefd));
    Invocation first = {.c = context()};
    fclose(first.c.input);
    first.c.input = fdopen(pipefd[0], "rb"); CHECK(first.c.input);
    first.c.isCancelled = cancelled; first.c.cancellation = &first.cancellation;
    const char *blocked[] = {"ffmpeg", "-v", "error", "-f", "s16le", "-ar", "8000",
        "-ac", "1", "-i", "pipe:0", "-f", "null", "-", NULL};
    first.args = blocked;
    pthread_t thread; CHECK(!pthread_create(&thread, NULL, invoke, &first));
    for (int i = 0; i < 500 && atomic_load(&first.cancellation.polls) < 3; ++i) usleep(10000);
    CHECK(atomic_load(&first.cancellation.polls) >= 3);

    // A queued caller can cancel without entering FFmpeg or changing the active
    // command's streams, globals or cancellation context.
    Invocation second = {.c = context(), .args = probe_version};
    second.c.isCancelled = cancelled; second.c.cancellation = &second.cancellation;
    pthread_t queued; CHECK(!pthread_create(&queued, NULL, invoke, &second));
    usleep(100000);
    CHECK(!atomic_load(&second.finished));
    atomic_store(&second.cancellation.cancelled, true);
    wait_flag(&second.finished);
    CHECK(!pthread_join(queued, NULL)); CHECK(second.status == 130);
    CHECK(!atomic_load(&first.finished));

    atomic_store(&first.cancellation.cancelled, true);
    wait_flag(&first.finished);
    CHECK(!pthread_join(thread, NULL)); CHECK(first.status == 130);
    close(pipefd[1]);
    close_context(&first.c); close_context(&second.c);

    // Also cancel after the scheduler has started its media workers, rather
    // than only while probing an input that has not produced its first byte.
    const char *live[] = {"ffmpeg", "-v", "error", "-re", "-f", "lavfi",
        "-i", "sine=sample_rate=8000", "-f", "null", "-", NULL};
    Invocation running = {.c = context(), .args = live};
    running.c.isCancelled = cancelled;
    running.c.cancellation = &running.cancellation;
    CHECK(!pthread_create(&thread, NULL, invoke, &running));
    usleep(200000);
    CHECK(!atomic_load(&running.finished));
    atomic_store(&running.cancellation.cancelled, true);
    wait_flag(&running.finished);
    CHECK(!pthread_join(thread, NULL)); CHECK(running.status == 130);
    close_context(&running.c);

    // Probe cancellation must cover both a blocked demuxer and a generated
    // input whose packet loop never performs a blocking read.
    const char *probe_blocked[] = {"ffprobe", "-v", "error", "-f", "s16le",
        "-ar", "8000", "-count_packets", "-show_streams", "pipe:0", NULL};
    const char *probe_live[] = {"ffprobe", "-v", "error", "-f", "lavfi",
        "-count_packets", "-show_streams", "sine=sample_rate=8000", NULL};
    for (int pass = 0; pass < 2; ++pass) {
        CHECK(!pipe(pipefd));
        Invocation probe = {.c = context(), .args = pass ? probe_live : probe_blocked};
        fclose(probe.c.input);
        probe.c.input = fdopen(pipefd[0], "rb"); CHECK(probe.c.input);
        probe.c.isCancelled = cancelled; probe.c.cancellation = &probe.cancellation;
        CHECK(!pthread_create(&thread, NULL, invoke, &probe));
        usleep(200000);
        if (atomic_load(&probe.finished)) {
            fprintf(stderr, "Probe cancellation case %d ended early with %d:\n", pass, probe.status);
            rewind(probe.c.error);
            int byte;
            while ((byte = fgetc(probe.c.error)) != EOF) fputc(byte, stderr);
        }
        CHECK(!atomic_load(&probe.finished));
        atomic_store(&probe.cancellation.cancelled, true);
        wait_flag(&probe.finished);
        CHECK(!pthread_join(thread, NULL)); CHECK(probe.status == 130);
        close(pipefd[1]); close_context(&probe.c);
    }
    expect(0, audio);
}

static void test_probe(void)
{
    FFmpegCommandContext c = context();
    CHECK(!run(&c, probe_version));
    rewind(c.output);
    char text[8192];
    size_t length = fread(text, 1, sizeof(text) - 1, c.output); text[length] = 0;
    CHECK(strstr(text, "ffprobe version 9.0.2"));
    close_context(&c);
    expect(0, (const char *[]){"ffprobe", "-h", NULL});
    expect(-1, (const char *[]){"ffprobe", "-not-a-real-option", NULL});
    expect(-1, probe_failure);
    expect(-1, (const char *[]){"ffprobe", "-v", "error", "-codec:a", "missing_codec",
        "tone.wav", NULL});
    // This reaches an upstream decoder-open error that previously called exit().
    expect(-1, (const char *[]){"ffprobe", "-v", "error", "-threads", "invalid",
        "tone.wav", NULL});
    c = context();
    CHECK(!run(&c, probe_audio));
    rewind(c.output);
    length = fread(text, 1, sizeof(text) - 1, c.output); text[length] = 0;
    CHECK(strstr(text, "\"codec_name\": \"pcm_s16le\""));
    CHECK(strstr(text, "\"sample_rate\": \"8000\""));
    CHECK(strstr(text, "\"duration\": \"0.100000\""));
    close_context(&c);

    // Counts, stream selections and formatter options must reset between calls.
    c = context();
    CHECK(!run(&c, (const char *[]){"ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
        "video.mkv", NULL}));
    rewind(c.output);
    CHECK(fgets(text, sizeof(text), c.output)); CHECK(!strcmp(text, "4\n"));
    close_context(&c);
    c = context();
    CHECK(!run(&c, (const char *[]){"ffprobe", "-v", "error", "tone.wav", NULL}));
    CHECK(ftell(c.output) == 0);
    close_context(&c);
    expect(0, (const char *[]){"ffprobe", "-v", "error", "-show_frames", "-show_log", "48",
        "-read_intervals", "%+#1", "-read_intervals", "%+#2", "tone.wav", NULL});
    expect(0, version);
}

static void test_external_codecs(void)
{
    const char *encoders[] = {"libx264", "libx265", "libvpx", "libvpx-vp9", "libsvtav1"};
    const char *option[] = {"-preset", "-x265-params", "-deadline", "-deadline", "-svtav1-params"};
    const char *value[] = {"ultrafast", "pools=none:frame-threads=1:log-level=error", "realtime", "realtime", "lp=1"};
    for (int i = 0; i < 5; ++i) {
        expect(0, (const char *[]){"ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=64x64:rate=10", "-frames:v", "2", "-threads", "1",
            "-c:v", encoders[i], option[i], value[i], "codec.mkv", NULL});
        FFmpegCommandContext c = context();
        // Explicitly use dav1d to validate the external AV1 decoder too.
        CHECK(!run(&c, (const char *[]){"ffmpeg", "-v", "error", "-threads", "1", "-c:v",
            i == 4 ? "libdav1d" : i == 1 ? "hevc" : i == 0 ? "h264" : i == 2 ? "vp8" : "vp9",
            "-i", "codec.mkv", "-threads", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1", NULL}));
        CHECK(ftell(c.output) == 2 * 64 * 64 * 3 / 2);
        close_context(&c);
    }
    for (int depth = 10; depth <= 12; depth += 2) {
        expect(0, (const char *[]){"ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=64x64:rate=10", "-frames:v", "1", "-threads", "1",
            "-pix_fmt", depth == 10 ? "yuv420p10le" : "yuv420p12le", "-c:v", "libx265",
            "-x265-params", "pools=none:frame-threads=1:log-level=error", "depth.mkv", NULL});
    }
    const char *audio_encoders[] = {"libopus", "libmp3lame"};
    for (int i = 0; i < 2; ++i) {
        expect(0, (const char *[]){"ffmpeg", "-v", "error", "-y", "-i", "tone.wav",
            "-ar", "48000", "-c:a", audio_encoders[i], "codec.mka", NULL});
        FFmpegCommandContext c = context();
        CHECK(!run(&c, (const char *[]){"ffmpeg", "-v", "error", "-i", "codec.mka",
            "-ar", "8000", "-f", "s16le", "pipe:1", NULL}));
        CHECK(ftell(c.output) >= 1600);
        close_context(&c);
    }
    double scores[2];
    for (int distorted = 0; distorted < 2; ++distorted) {
        expect(0, (const char *[]){"ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=192x108:rate=10:duration=0.2", "-filter_complex", distorted ?
            "split[a][b];[a]negate[d];[d][b]libvmaf=log_path=vmaf.json:log_fmt=json:n_threads=1" :
            "split[a][b];[a][b]libvmaf=log_path=vmaf.json:log_fmt=json:n_threads=1",
            "-f", "null", "-", NULL});
        char path[PATH_MAX]; snprintf(path, sizeof(path), "%s/vmaf.json", directory);
        FILE *report = fopen(path, "r"); CHECK(report);
        char json[16384]; size_t length = fread(json, 1, sizeof(json) - 1, report); json[length] = 0;
        fclose(report);
        char *score = strstr(json, "\"vmaf\":"); CHECK(score);
        scores[distorted] = strtod(score + strlen("\"vmaf\":"), NULL);
    }
    // Identical frames need not score 100 (the first frame has zero motion).
    CHECK(scores[0] > 95.0 && scores[0] <= 100.0);
    CHECK(scores[1] >= 0 && scores[1] < scores[0] - 20.0);
    for (int i = 0; i < 2; ++i)
        expect(-1, (const char *[]){"ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=64x64:rate=10", "-frames:v", "1", "-c:v", encoders[i],
            "-preset", "invalid-preset", "-f", "null", "-", NULL});
    expect(-1, (const char *[]){"ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=192x108:rate=10:duration=0.2", "-filter_complex",
        "split[a][b];[a][b]libvmaf=model=path=missing-model.json", "-f", "null", "-", NULL});
}

// Lossless modes provide a pixel oracle independent of file size or codec tags.
static void test_lossless_and_damaged_input(void)
{
    unsigned char reference[4 * 32 * 32 * 3 / 2], decoded[sizeof(reference)];
    FFmpegCommandContext c = context();
    const char *decode[] = {"ffmpeg", "-v", "error", "-threads", "1", "-i", "video.mkv",
        "-threads", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1", NULL};
    CHECK(!run(&c, decode)); rewind(c.output);
    CHECK(fread(reference, 1, sizeof(reference), c.output) == sizeof(reference));
    close_context(&c);
    const char *encoders[] = {"libx264", "libx265", "libvpx-vp9"};
    const char *options[] = {"-crf", "-x265-params", "-lossless"};
    const char *values[] = {"0", "lossless=1:pools=none:frame-threads=1:log-level=error", "1"};
    for (int i = 0; i < 3; ++i) {
        expect(0, (const char *[]){"ffmpeg", "-v", "error", "-y", "-i", "video.mkv",
            "-threads", "1", "-c:v", encoders[i], options[i], values[i], "lossless.mkv", NULL});
        c = context();
        CHECK(!run(&c, (const char *[]){"ffmpeg", "-v", "error", "-threads", "1", "-i", "lossless.mkv",
            "-threads", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1", NULL}));
        rewind(c.output);
        CHECK(fread(decoded, 1, sizeof(decoded), c.output) == sizeof(decoded));
        CHECK(fgetc(c.output) == EOF);
        CHECK(!memcmp(reference, decoded, sizeof(reference)));
        close_context(&c);
    }
    char path[PATH_MAX]; snprintf(path, sizeof(path), "%s/lossless.mkv", directory);
    FILE *source = fopen(path, "rb"); CHECK(source);
    unsigned char header[16]; CHECK(fread(header, 1, sizeof(header), source) == sizeof(header));
    fclose(source);
    for (int bytes = 0; bytes <= 16; bytes += 4) {
        snprintf(path, sizeof(path), "%s/damaged.mkv", directory);
        FILE *damaged = fopen(path, "wb"); CHECK(damaged);
        CHECK(fwrite(header, 1, bytes, damaged) == bytes); CHECK(!fclose(damaged));
        expect(-1, (const char *[]){"ffmpeg", "-v", "error", "-i", "damaged.mkv", "-f", "null", "-", NULL});
        expect(-1, (const char *[]){"ffprobe", "-v", "error", "-show_streams", "damaged.mkv", NULL});
    }
    expect(0, probe_audio);
}

static void test_codec_sessions(void)
{
    char paths[2][PATH_MAX], cwd[PATH_MAX], after[PATH_MAX];
    CHECK(getcwd(cwd, sizeof(cwd)));
    for (int i = 0; i < 2; ++i) {
        snprintf(paths[i], sizeof(paths[i]), "%s/session %c", directory, 'A' + i);
        CHECK(!mkdir(paths[i], 0700) || errno == EEXIST);
    }
    // Interleave two passes with identical relative filenames in two sessions.
    // Pass 2 must read that session's statistics, including encoder sidecars.
    for (int pass = 1; pass <= 2; ++pass) for (int session = 0; session < 2; ++session) {
        FFmpegCommandContext c = context(); c.directory = paths[session]; c.home = paths[session];
        CHECK(!run(&c, (const char *[]){"ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            session ? "testsrc2=size=96x64:rate=10" : "testsrc2=size=64x64:rate=10",
            "-frames:v", "6", "-threads", "1", "-c:v", "libx264", "-b:v", "100k",
            "-pass", pass == 1 ? "1" : "2", "-passlogfile", "stats", "out.mkv", NULL}));
        close_context(&c);
        char path[PATH_MAX]; snprintf(path, sizeof(path), "%s/stats-0.log", paths[session]);
        struct stat st; CHECK(!stat(path, &st) && st.st_size > 0);
    }
    for (int session = 0; session < 2; ++session) {
        FFmpegCommandContext c = context(); c.directory = paths[session]; c.home = paths[session];
        CHECK(!run(&c, (const char *[]){"ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width", "-of", "csv=p=0", "/home/out.mkv", NULL}));
        rewind(c.output); char width[32]; CHECK(fgets(width, sizeof(width), c.output));
        CHECK(!strcmp(width, session ? "96\n" : "64\n")); close_context(&c);
    }
    // Close every stream between commands: SVT must not cache the first FILE*
    // or the first session's logging environment.
    const char *noisy[] = {"SVT_LOG=3", "AV_LOG_FORCE_NOCOLOR=1", NULL};
    const char *quiet[] = {"SVT_LOG=-1", "AV_LOG_FORCE_NOCOLOR=1", NULL};
    for (int i = 0; i < 3; ++i) {
        FFmpegCommandContext c = context(); c.directory = paths[i % 2];
        c.environment = i == 1 ? quiet : noisy;
        CHECK(!run(&c, (const char *[]){"ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=64x64:rate=10", "-frames:v", "1", "-c:v", "libsvtav1",
            "-preset", "12", "-svtav1-params", "lp=1", "-f", "null", "-", NULL}));
        CHECK(i == 1 ? ftell(c.error) == 0 : ftell(c.error) > 0);
        close_context(&c);
    }
    CHECK(getcwd(after, sizeof(after))); CHECK(!strcmp(cwd, after));
}

static void test_encoder_cancellation(void)
{
    const char *encoders[] = {"libx264", "libx265", "libvpx", "libvpx-vp9", "libsvtav1"};
    const char *option[] = {"-preset", "-x265-params", "-deadline", "-deadline", "-svtav1-params"};
    const char *value[] = {"ultrafast", "pools=none:frame-threads=1:log-level=error", "realtime", "realtime", "lp=1"};
    for (int codec = 0; codec < 5; ++codec) {
        char progress[PATH_MAX]; snprintf(progress, sizeof(progress), "%s/encoder-progress.txt", directory);
        unlink(progress);
        const char *args[] = {"ffmpeg", "-v", "error", "-re", "-f", "lavfi", "-i",
            "testsrc2=size=64x64:rate=30", "-threads", "1", "-c:v", encoders[codec],
            option[codec], value[codec], "-stats_period", "0.01", "-progress", "encoder-progress.txt",
            "-f", "null", "-", NULL};
        Invocation running = {.c = context(), .args = args};
        running.c.isCancelled = cancelled; running.c.cancellation = &running.cancellation;
        pthread_t worker; CHECK(!pthread_create(&worker, NULL, invoke, &running));
        int frames = 0;
        // Wait for actual encoded output, not a timing guess about initialization.
        for (int poll = 0; poll < 1000 && !frames; ++poll) {
            FILE *file = fopen(progress, "r");
            if (file) {
                char line[256];
                while (fgets(line, sizeof(line), file)) {
                    int count;
                    if (sscanf(line, "frame=%d", &count) == 1 && count > 0) frames = count;
                }
                fclose(file);
            }
            CHECK(!atomic_load(&running.finished));
            if (!frames) usleep(10000);
        }
        CHECK(frames > 0);
        Invocation queued = {.c = context(), .args = probe_version};
        queued.c.isCancelled = cancelled; queued.c.cancellation = &queued.cancellation;
        pthread_t waiting; CHECK(!pthread_create(&waiting, NULL, invoke, &queued));
        for (int poll = 0; poll < 500 && !atomic_load(&queued.cancellation.polls); ++poll) usleep(10000);
        CHECK(atomic_load(&queued.cancellation.polls)); CHECK(!atomic_load(&queued.finished));
        atomic_store(&running.cancellation.cancelled, true);
        wait_flag(&running.finished); CHECK(!pthread_join(worker, NULL)); CHECK(running.status == 130);
        wait_flag(&queued.finished); CHECK(!pthread_join(waiting, NULL)); CHECK(queued.status == 0);
        rewind(queued.c.output); char line[256]; CHECK(fgets(line, sizeof(line), queued.c.output));
        CHECK(strstr(line, "ffprobe version"));
        close_context(&running.c); close_context(&queued.c);
    }
    expect(0, audio);
}

int main(void)
{
    const char *base = getenv("TEST_TMPDIR");
    if (!base) base = "/tmp";
    snprintf(directory, sizeof(directory), "%s/ffmpeg-tests.XXXXXX", base);
    CHECK(mkdtemp(directory));
    char resolved[PATH_MAX]; CHECK(realpath(directory, resolved));
    strcpy(directory, resolved);
    const int signals[] = {SIGINT, SIGTERM, SIGQUIT, SIGPIPE, SIGXCPU};
    struct sigaction original[5];
    for (int i = 0; i < 5; ++i) CHECK(!sigaction(signals[i], NULL, &original[i]));

    expect(0, version);
    expect(0, (const char *[]){"ffmpeg", "-h", "encoder=pcm_s16le", NULL});
    expect(-1, (const char *[]){"ffmpeg", "-not-a-real-option", NULL});
    expect(0, audio);
    expect(-1, (const char *[]){"ffmpeg", "-v", "error", "-max_alloc", "32",
        "-i", "tone.wav", "-f", "null", "-", NULL});
    // The private libav allocation limit must return to its default next run.
    expect(0, audio);
    expect(0, (const char *[]){"ffmpeg", "-v", "error", "-i", "tone.wav",
        "-threads", "2", "-f", "null", "-", NULL});

    // -y from the preceding command must not survive. Upstream reports the
    // overwrite refusal as AVERROR_EXIT; assert that the file stayed untouched.
    char tone[PATH_MAX]; snprintf(tone, sizeof(tone), "%s/tone.wav", directory);
    struct stat before, after; CHECK(!stat(tone, &before));
    expect(0, (const char *[]){"ffmpeg", "-v", "error", "-f", "lavfi",
        "-i", "sine=frequency=220:sample_rate=8000", "-t", "1", "tone.wav", NULL});
    CHECK(!stat(tone, &after)); CHECK(before.st_size == after.st_size);
    expect(0, audio);

    // Media output must go to the command descriptor, with exact sample count.
    FFmpegCommandContext c = context();
    CHECK(!run(&c, (const char *[]){"ffmpeg", "-v", "error", "-i", "/home/tone.wav",
        "-f", "s16le", "pipe:1", NULL}));
    CHECK(ftell(c.output) == 1600);
    // Feed the same media back through command stdin.
    rewind(c.output);
    FILE *swap = c.input; c.input = c.output; c.output = swap;
    CHECK(!run(&c, (const char *[]){"ffmpeg", "-v", "error", "-f", "s16le", "-ar", "8000",
        "-ac", "1", "-i", "pipe:0", "-f", "s16le", "pipe:1", NULL}));
    CHECK(ftell(c.output) == 1600);
    close_context(&c);

    // fd: keeps the seekability of the resolved command stream, allowing the
    // WAV muxer to seek back and finalize the RIFF length.
    c = context();
    CHECK(!run(&c, (const char *[]){"ffmpeg", "-v", "error", "-i", "tone.wav",
        "-f", "wav", "fd:", NULL}));
    long wav_size = ftell(c.output);
    CHECK(wav_size > 1600);
    CHECK(!fseek(c.output, 4, SEEK_SET));
    unsigned char riff_size[4];
    CHECK(fread(riff_size, 1, 4, c.output) == 4);
    CHECK(((unsigned)riff_size[0] | (unsigned)riff_size[1] << 8 |
        (unsigned)riff_size[2] << 16 | (unsigned)riff_size[3] << 24) == wav_size - 8);
    close_context(&c);

    expect(0, (const char *[]){"ffmpeg", "-v", "error", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=32x32:rate=10", "-frames:v", "4", "-threads", "2",
        "-c:v", "ffv1", "video.mkv", NULL});
    c = context();
    CHECK(!run(&c, (const char *[]){"ffmpeg", "-v", "error", "-i", "video.mkv",
        "-threads", "2", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1", NULL}));
    CHECK(ftell(c.output) == 4 * 32 * 32 * 3 / 2);
    close_context(&c);
    test_probe();
    test_cancellation();
    test_external_codecs();
    test_lossless_and_damaged_input();
    test_codec_sessions();
    test_encoder_cancellation();

    // A closed downstream reader must return an error without delivering a
    // process-wide SIGPIPE or leaving workers behind.
    int broken[2]; CHECK(!pipe(broken)); close(broken[0]);
    c = context(); fclose(c.output); c.output = fdopen(broken[1], "wb"); CHECK(c.output);
    CHECK(run(&c, (const char *[]){"ffmpeg", "-v", "error", "-i", "tone.wav",
        "-f", "s16le", "pipe:1", NULL}) != 0);
    close_context(&c);

    // Help/version uses stdio rather than libavformat's pipe writer, and must
    // likewise survive a closed reader without changing descriptor flags.
    CHECK(!pipe(broken)); close(broken[0]);
    c = context(); fclose(c.output); c.output = fdopen(broken[1], "wb"); CHECK(c.output);
    int no_sigpipe = fcntl(broken[1], F_GETNOSIGPIPE);
    CHECK(run(&c, version) != 0);
    CHECK(fcntl(broken[1], F_GETNOSIGPIPE) == no_sigpipe);
    // The failed stdio write legitimately leaves the caller's error indicator.
    clearerr(c.output);
    close_context(&c);

    const char *reports[] = {"FFREPORT=file=report.log:level=32", "AV_LOG_FORCE_NOCOLOR=1", NULL};
    unsigned report_fds = descriptors();
    c = context(); c.environment = reports;
    CHECK(!run(&c, version)); close_context(&c);
    expect(0, version);
    CHECK(descriptors() == report_fds);

    // Warm allocations first; compare live allocations, not allocator RSS caches.
    for (int i = 0; i < 5; ++i) {
        expect(0, audio); expect(-1, failure);
        expect(0, probe_audio); expect(-1, probe_failure); test_probe();
    }
    unsigned initial_fds = descriptors();
    unsigned initial_threads = threads();
    size_t initial_allocated = allocated_bytes();
    int iterations = getenv("FFMPEG_TEST_SOAK") ? 200 : 40;
    for (int i = 0; i < iterations; ++i) {
        expect(0, version); expect(-1, failure); expect(0, audio);
        expect(0, probe_audio); expect(-1, probe_failure); test_probe();
        if (i % 10 == 0) {
            test_external_codecs();
            test_lossless_and_damaged_input();
            test_codec_sessions();
        }
        if (i == 20) test_encoder_cancellation();
    }
    size_t final_allocated = allocated_bytes();
    CHECK(descriptors() == initial_fds);
    CHECK(threads() == initial_threads);
#if __has_feature(address_sanitizer)
    // ASan reports exact live allocations; do not hide per-invocation leaks
    // behind the allowance needed by the native allocator's accounting.
    CHECK(final_allocated <= initial_allocated);
#else
    CHECK(final_allocated <= initial_allocated + 1024 * 1024);
#endif
    for (int i = 0; i < 5; ++i) {
        struct sigaction current; CHECK(!sigaction(signals[i], NULL, &current));
        CHECK(current.sa_handler == original[i].sa_handler);
    }
    fprintf(stderr, "FFmpeg lifecycle, media, cancellation and serialization tests passed (%d iterations); live allocation delta: %lld bytes\n",
        iterations, (long long)final_allocated - (long long)initial_allocated);
    return 0;
}
