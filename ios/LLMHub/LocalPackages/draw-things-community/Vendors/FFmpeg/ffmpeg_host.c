#include "ffmpeg_host.h"
#include "config.h"
#if CONFIG_LIBSVTAV1
#include <svt-av1/EbSvtAv1Enc.h>
#endif

#include <errno.h>
#include <limits.h>
#include <poll.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

extern int ffmpeg_command_main(int argc, char **argv);
extern int ffprobe_command_main(int argc, char **argv);
extern void ffmpeg_show_help_default(const char *opt, const char *arg);
extern void ffprobe_show_help_default(const char *opt, const char *arg);
extern const char *program_name;
extern int program_birth_year;
static pthread_mutex_t command_mutex = PTHREAD_MUTEX_INITIALIZER;
static const FFmpegCommandContext *active;
static void (*command_help)(const char *, const char *);

#if CONFIG_LIBSVTAV1
static void svt_log(void *context, SvtAv1LogLevel level, const char *tag,
    const char *format, va_list args)
{
    // SVT's default global logger caches a FILE* from its first invocation.
    // Install a context-free callback once and resolve the active stream here.
    const char *configured = ffmpeg_host_getenv("SVT_LOG");
    int threshold = configured ? atoi(configured) : SVT_AV1_LOG_INFO;
    if (level > threshold) return;
    FILE *stream = ffmpeg_host_stream(2);
    if (tag) fprintf(stream, "%s: ", tag);
    vfprintf(stream, format, args);
}
#endif

// Shared upstream help dispatch calls the current frontend's help function.
void show_help_default(const char *opt, const char *arg)
{
    command_help(opt, arg);
}

int ffmpeg_host_cancelled(void)
{
    return active && active->isCancelled && active->isCancelled(active->cancellation);
}

static int command_run(int argc, char **argv, const FFmpegCommandContext *context,
    int (*command_main)(int, char **))
{
    if (argc < 1 || !argv || !context || !context->input || !context->output ||
        !context->error || !context->directory || context->directory[0] != '/')
        return 1;
    int previous;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &previous);
    int lock_status;
    while ((lock_status = pthread_mutex_trylock(&command_mutex)) == EBUSY) {
        if (context->isCancelled && context->isCancelled(context->cancellation)) {
            pthread_setcancelstate(previous, NULL);
            return 130;
        }
        const struct timespec delay = {0, 10000000};
        nanosleep(&delay, NULL);
    }
    if (lock_status) {
        pthread_setcancelstate(previous, NULL);
        return 1;
    }
    active = context;
    int is_probe = command_main == ffprobe_command_main;
    program_name = is_probe ? "ffprobe" : "ffmpeg";
    program_birth_year = is_probe ? 2007 : 2000;
    command_help = is_probe ? ffprobe_show_help_default : ffmpeg_show_help_default;
#if CONFIG_LIBSVTAV1
    static int svt_logger_initialized;
    if (!svt_logger_initialized) {
        svt_av1_set_log_callback(svt_log, NULL);
        svt_logger_initialized = 1;
    }
#endif
    // stdio diagnostics/help also write to command pipes. Suppress SIGPIPE on
    // these descriptors for the invocation, then restore the host's flags.
    int output_fds[] = {fileno(context->output), fileno(context->error)};
    int no_sigpipe[2];
    for (int i = 0; i < 2; ++i) {
        no_sigpipe[i] = fcntl(output_fds[i], F_GETNOSIGPIPE);
        if (no_sigpipe[i] == 0) fcntl(output_fds[i], F_SETNOSIGPIPE, 1);
    }
    int status = ffmpeg_host_cancelled() ? 130 : command_main(argc, argv);
    if (ffmpeg_host_cancelled()) status = 130;
    if (fflush(context->output) || ferror(context->output)) {
        // Darwin stdio can retain unwritten bytes after EPIPE. Discard that
        // failed output before restoring the descriptor's SIGPIPE behavior,
        // otherwise the host's later fclose could retry it and raise SIGPIPE.
        fpurge(context->output);
        if (!status) status = 1;
    }
    if (fflush(context->error)) fpurge(context->error);
    for (int i = 1; i >= 0; --i)
        if (no_sigpipe[i] == 0) fcntl(output_fds[i], F_SETNOSIGPIPE, 0);
    active = NULL;
    pthread_mutex_unlock(&command_mutex);
    pthread_setcancelstate(previous, NULL);
    return status < 0 ? 1 : status;
}

int FFmpegCommandRun(int argc, char **argv, const FFmpegCommandContext *context)
{
    return command_run(argc, argv, context, ffmpeg_command_main);
}

int FFprobeCommandRun(int argc, char **argv, const FFmpegCommandContext *context)
{
    return command_run(argc, argv, context, ffprobe_command_main);
}

FILE *ffmpeg_host_stream(int fd)
{
    if (!active) return fd == 0 ? stdin : fd == 1 ? stdout : stderr;
    return fd == 0 ? active->input : fd == 1 ? active->output : active->error;
}

void ffmpeg_host_perror(const char *message)
{
    int error = errno;
    fprintf(ffmpeg_host_stream(2), "%s%s%s\n", message ? message : "",
        message && *message ? ": " : "", strerror(error));
}

int ffmpeg_host_descriptor(int fd)
{
    // Numeric descriptors beyond the standard streams are real host descriptors.
    return fd >= 0 && fd <= 2 ? fileno(ffmpeg_host_stream(fd)) : fd;
}

char *ffmpeg_host_getenv(const char *name)
{
    if (!active || !active->environment) return NULL;
    size_t length = strlen(name);
    for (const char *const *entry = active->environment; *entry; ++entry)
        if (!strncmp(*entry, name, length) && (*entry)[length] == '=')
            return (char *)*entry + length + 1;
    return NULL;
}

const char *ffmpeg_host_temporary_directory(void)
{
    return active->temporaryDirectory ? active->temporaryDirectory : active->directory;
}

static const char *command_path(const char *path, char buffer[PATH_MAX])
{
    if (!active || !path || !*path) return path;
    const char *prefix = NULL;
    const char *suffix = path;
    if (*path != '/') {
        prefix = active->directory;
    } else {
        const char *aliases[] = {"/home", "/tmp", "/app_bundle"};
        const char *targets[] = {active->home, active->temporaryDirectory, active->appBundle};
        for (int i = 0; i < 3; ++i) {
            size_t length = strlen(aliases[i]);
            if (targets[i] && !strncmp(path, aliases[i], length) &&
                (path[length] == '/' || path[length] == '\0')) {
                prefix = targets[i];
                suffix = path + length;
                break;
            }
        }
    }
    if (!prefix) return path;
    int length = snprintf(buffer, PATH_MAX, "%s%s%s", prefix,
        *suffix && *suffix != '/' ? "/" : "", suffix);
    if (length < 0 || length >= PATH_MAX) {
        errno = ENAMETOOLONG;
        return NULL;
    }
    return buffer;
}

int ffmpeg_host_open(const char *path, int flags, ...)
{
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, int);
        va_end(args);
    }
    char buffer[PATH_MAX];
    path = command_path(path, buffer);
    return path ? open(path, flags, mode) : -1;
}

FILE *ffmpeg_host_fopen(const char *path, const char *mode)
{
    char buffer[PATH_MAX];
    path = command_path(path, buffer);
    return path ? fopen(path, mode) : NULL;
}

#define PATH_OPERATION(name, declaration, call) \
    declaration { \
        char buffer[PATH_MAX]; \
        path = command_path(path, buffer); \
        return path ? call : -1; \
    }
PATH_OPERATION(stat, int ffmpeg_host_stat(const char *path, struct stat *st), stat(path, st))
PATH_OPERATION(lstat, int ffmpeg_host_lstat(const char *path, struct stat *st), lstat(path, st))
PATH_OPERATION(access, int ffmpeg_host_access(const char *path, int mode), access(path, mode))
PATH_OPERATION(unlink, int ffmpeg_host_unlink(const char *path), unlink(path))
PATH_OPERATION(mkdir, int ffmpeg_host_mkdir(const char *path, mode_t mode), mkdir(path, mode))
PATH_OPERATION(rmdir, int ffmpeg_host_rmdir(const char *path), rmdir(path))

int ffmpeg_host_rename(const char *from, const char *to)
{
    char source[PATH_MAX], destination[PATH_MAX];
    from = command_path(from, source);
    to = command_path(to, destination);
    return from && to ? rename(from, to) : -1;
}

DIR *ffmpeg_host_opendir(const char *path)
{
    char buffer[PATH_MAX];
    path = command_path(path, buffer);
    return path ? opendir(path) : NULL;
}

int ffmpeg_host_glob(const char *pattern, int flags,
    int (*errfunc)(const char *, int), glob_t *result)
{
    char buffer[PATH_MAX];
    pattern = command_path(pattern, buffer);
    return pattern ? glob(pattern, flags, errfunc, result) : GLOB_ABORTED;
}

static int wait_descriptor(int fd, short events)
{
    struct pollfd descriptor = {fd, events, 0};
    for (;;) {
        if (ffmpeg_host_cancelled()) {
            errno = EINTR;
            return -1;
        }
        int result = poll(&descriptor, 1, 50);
        if (result > 0) return 0;
        if (result < 0 && errno != EINTR) return -1;
    }
}

ssize_t ffmpeg_host_read(int fd, void *buffer, size_t size)
{
    if (wait_descriptor(fd, POLLIN)) return -1;
    return read(fd, buffer, size);
}

ssize_t ffmpeg_host_write(int fd, const void *buffer, size_t size)
{
    if (wait_descriptor(fd, POLLOUT)) return -1;
    struct stat st;
    if (fstat(fd, &st) < 0) return -1;
    if (S_ISFIFO(st.st_mode) || S_ISSOCK(st.st_mode)) {
        // Suppress SIGPIPE on this descriptor, never process-wide. Keep writes
        // small so a stalled reader cannot strand worker teardown.
        if (fcntl(fd, F_SETNOSIGPIPE, 1) < 0) return -1;
        if (size > PIPE_BUF) size = PIPE_BUF;
    }
    return write(fd, buffer, size);
}
