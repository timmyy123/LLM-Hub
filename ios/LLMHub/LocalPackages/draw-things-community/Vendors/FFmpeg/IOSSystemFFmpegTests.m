#import <Foundation/Foundation.h>
#import "ios_system/ios_system.h"
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <signal.h>

#define CHECK(condition) do { if (!(condition)) { \
    dprintf(STDERR_FILENO, "FAIL line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)

static atomic_bool finished;
static int cancelledStatus;
static bool cancelProbe;
static volatile sig_atomic_t receivedSignal;
static void signalHandler(int signal) { receivedSignal = signal; }
static void *runCancelled(void *opaque)
{
    @autoreleasepool {
        ios_switchSession("ffmpeg-cancel-test");
        FILE **streams = opaque;
        ios_setStreams(streams[0], streams[1], streams[2]);
        const char *args[] = {"ffmpeg", "-v", "error", "-f", "s16le", "-ar", "8000",
            "-ac", "1", "-i", "pipe:0", "-f", "null", "-"};
        cancelledStatus = cancelProbe
            ? ios_system_osh("ffprobe -v error -f lavfi -count_packets -show_streams sine=sample_rate=8000")
            : ios_system(sizeof(args) / sizeof(args[0]), args);
        atomic_store(&finished, true);
    }
    return NULL;
}

int main(void)
{
    @autoreleasepool {
        initializeEnvironment();
        CHECK(ios_executable("ffmpeg"));
        CHECK(ios_executable("ffprobe"));
        CHECK(ios_commandIsExclusiveOrThreadSafe("/usr/bin/ffmpeg") ==
            IOS_COMMAND_ISOLATION_EXCLUSIVE_FFMPEG);
        CHECK(ios_commandIsExclusiveOrThreadSafe("/usr/bin/ffprobe") ==
            IOS_COMMAND_ISOLATION_EXCLUSIVE_FFMPEG);
        NSFileManager *fm = NSFileManager.defaultManager;
        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:NSUUID.UUID.UUIDString] isDirectory:YES];
        NSURL *project = [root URLByAppendingPathComponent:@"Project" isDirectory:YES];
        CHECK([fm createDirectoryAtURL:project withIntermediateDirectories:YES attributes:nil error:nil]);
        ios_setStreams(stdin, stdout, stderr);
        CHECK(ios_setSystemPaths(project, root, root, root, project, root));
        ios_setDirectoryURL(project);
        CHECK(ios_system_osh(
            "ffmpeg -version > version.txt; "
            "ffmpeg -v error -y -f lavfi -i sine=sample_rate=8000 -t 0.1 tone.wav; "
            "ffmpeg -v error -i tone.wav -f s16le pipe:1 | "
            "ffmpeg -v error -f s16le -ar 8000 -ac 1 -i pipe:0 -f s16le pipe:1 > samples.raw") == 0);
        NSData *samples = [NSData dataWithContentsOfURL:[project URLByAppendingPathComponent:@"samples.raw"]];
        CHECK(samples.length == 1600);
        NSString *version = [NSString stringWithContentsOfURL:[project URLByAppendingPathComponent:@"version.txt"]
            encoding:NSUTF8StringEncoding error:nil];
        CHECK([version containsString:@"ffmpeg version 9.0.2"]);
        CHECK(ios_system_osh("ffprobe -v error -select_streams a:0 "
            "-show_entries stream=codec_name,sample_rate:format=duration "
            "-of json /home/Project/tone.wav > probe.json") == 0);
        NSData *probeData = [NSData dataWithContentsOfURL:[project URLByAppendingPathComponent:@"probe.json"]];
        CHECK(probeData);
        NSDictionary *probe = [NSJSONSerialization JSONObjectWithData:probeData options:0 error:nil];
        CHECK(probe && [probe[@"streams"] count] == 1);
        CHECK([probe[@"streams"][0][@"codec_name"] isEqualToString:@"pcm_s16le"]);
        CHECK([probe[@"streams"][0][@"sample_rate"] isEqualToString:@"8000"]);
        CHECK([probe[@"format"][@"duration"] isEqualToString:@"0.100000"]);
        CHECK(ios_system_osh("ffmpeg -v error -i tone.wav -f wav pipe:1 | "
            "ffprobe -v error -show_entries stream=codec_name -of csv=p=0 - > probe-pipe.txt") == 0);
        NSString *probePipe = [NSString stringWithContentsOfURL:[project URLByAppendingPathComponent:@"probe-pipe.txt"]
            encoding:NSUTF8StringEncoding error:nil];
        CHECK([probePipe isEqualToString:@"pcm_s16le\n"]);
        CHECK(ios_system_osh("ffprobe -v error -codec:a nonexistent tone.wav") != 0);
        CHECK(ios_system_osh("ffprobe -v error -show_format tone.wav > /dev/null") == 0);
        CHECK(ios_system_osh("FFREPORT=file=session-report.log:level=32 ffmpeg -version > /dev/null") == 0);
        CHECK([fm fileExistsAtPath:[project URLByAppendingPathComponent:@"session-report.log"].path]);
        CHECK(ios_system_osh("ffmpeg -v error -i /home/Project/tone.wav -f null -") == 0);
        CHECK(ios_system_osh("ffmpeg -v error -i missing.wav -f null -") != 0);
        CHECK(ios_system_osh("ffmpeg -v error -i tone.wav -f null -") == 0);

        // Exercise every external library through OSH, including quoted paths,
        // command-local environment, redirection, and real ffprobe JSON parsing.
        CHECK(ios_system_osh("ffmpeg -v error -y -f lavfi -i testsrc2=size=64x64:rate=10 "
            "-i tone.wav -frames:v 2 -threads 1 -c:v libx264 -preset ultrafast "
            "-c:a libopus -ar 48000 'codec clip.mkv'") == 0);
        CHECK(ios_system_osh("ffmpeg -v error -y -i 'codec clip.mkv' -threads 1 "
            "-c:v libx265 -pix_fmt yuv420p10le "
            "-x265-params pools=none:frame-threads=1:log-level=error -c:a libmp3lame hevc.mkv") == 0);
        CHECK(ios_system_osh("ffmpeg -v error -y -i hevc.mkv -an -threads 1 "
            "-c:v libvpx-vp9 -deadline realtime -pix_fmt yuv420p vp9.webm") == 0);
        CHECK(ios_system_osh("SVT_LOG=-1 ffmpeg -v error -y -i vp9.webm -an "
            "-c:v libsvtav1 -preset 12 -svtav1-params lp=1 av1.mkv 2> svt.log") == 0);
        CHECK(ios_system_osh("ffmpeg -v error -c:v libdav1d -i av1.mkv -threads 1 "
            "-pix_fmt yuv420p -f rawvideo pipe:1 > av1.raw") == 0);
        CHECK([NSData dataWithContentsOfURL:[project URLByAppendingPathComponent:@"av1.raw"]].length
            == 2 * 64 * 64 * 3 / 2);
        CHECK([NSData dataWithContentsOfURL:[project URLByAppendingPathComponent:@"svt.log"]].length == 0);
        CHECK(ios_system_osh("ffmpeg -v error -f lavfi -i testsrc2=size=192x108:rate=10:duration=0.2 "
            "-filter_complex 'split[a][b];[a][b]libvmaf=log_path=vmaf.json:log_fmt=json:n_threads=1' "
            "-f null -") == 0);
        NSData *vmafData = [NSData dataWithContentsOfURL:[project URLByAppendingPathComponent:@"vmaf.json"]];
        CHECK(vmafData);
        NSDictionary *vmaf = [NSJSONSerialization JSONObjectWithData:vmafData options:0 error:nil];
        CHECK([vmaf[@"frames"] count] == 2);
        CHECK([vmaf[@"pooled_metrics"][@"vmaf"][@"mean"] doubleValue] > 95);
        CHECK(ios_system_osh("ffprobe -v error -show_streams -of json hevc.mkv > codecs.json") == 0);
        NSData *codecsData = [NSData dataWithContentsOfURL:[project URLByAppendingPathComponent:@"codecs.json"]];
        CHECK(codecsData);
        NSDictionary *codecs = [NSJSONSerialization JSONObjectWithData:codecsData options:0 error:nil];
        CHECK([codecs[@"streams"] count] == 2);
        CHECK([codecs[@"streams"][0][@"codec_name"] isEqualToString:@"hevc"]);
        CHECK([codecs[@"streams"][0][@"pix_fmt"] isEqualToString:@"yuv420p10le"]);
        CHECK([codecs[@"streams"][1][@"codec_name"] isEqualToString:@"mp3"]);

        // ios_kill must use the cooperative token without calling an unrelated
        // process SIGINT handler or cancelling FFmpeg's command pthread.
        struct sigaction handler = {.sa_handler = signalHandler}, savedHandler;
        sigemptyset(&handler.sa_mask);
        CHECK(!sigaction(SIGINT, &handler, &savedHandler));
        for (int pass = 0; pass < 2; ++pass) {
            cancelProbe = pass != 0;
            atomic_store(&finished, false);
            ios_switchSession("ffmpeg-cancel-test");
            ios_setDirectoryURL(project);
            int pipefd[2]; CHECK(!pipe(pipefd));
            // OSH buffers stdin before dispatch; use EOF and an endless lavfi
            // source to exercise cancellation of the actual FFprobe command.
            if (cancelProbe) { close(pipefd[1]); pipefd[1] = -1; }
            FILE *input = fdopen(pipefd[0], "rb"), *output = tmpfile(), *error = tmpfile();
            CHECK(input && output && error);
            ios_setStreams(input, output, error);
            FILE *streams[] = {input, output, error};
            pthread_t worker; CHECK(!pthread_create(&worker, NULL, runCancelled, streams));
            usleep(500000);
            CHECK(!atomic_load(&finished));
            CHECK(ios_kill() == 0);
            for (int i = 0; i < 500 && !atomic_load(&finished); ++i) usleep(10000);
            CHECK(atomic_load(&finished));
            CHECK(!pthread_join(worker, NULL));
            CHECK(cancelledStatus != 0);
            CHECK(receivedSignal == 0);
            if (pipefd[1] >= 0) close(pipefd[1]);
            ios_setStreams(stdin, stdout, stderr);
            fclose(input); fclose(output); fclose(error);
            CHECK(ios_system_osh("ffmpeg -v error -i tone.wav -f null -") == 0);
            ios_closeSession("ffmpeg-cancel-test");
        }
        CHECK(!sigaction(SIGINT, &savedHandler, NULL));
        [fm removeItemAtURL:root error:nil];
        fprintf(stderr, "FFmpeg ios_system registry, OSH pipes, environment, paths and cancellation tests passed.\n");
        return 0;
    }
}
