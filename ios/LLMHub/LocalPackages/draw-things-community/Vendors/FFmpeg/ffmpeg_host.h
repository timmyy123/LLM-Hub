#ifndef DRAW_THINGS_FFMPEG_HOST_H
#define DRAW_THINGS_FFMPEG_HOST_H

#include "FFmpegCommand.h"
#include <dirent.h>
#include <fcntl.h>
#include <glob.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/stat.h>

FILE *ffmpeg_host_stream(int fd);
void ffmpeg_host_perror(const char *message);
char *ffmpeg_host_getenv(const char *name);
const char *ffmpeg_host_temporary_directory(void);
int ffmpeg_host_cancelled(void);
int ffmpeg_command_main(int argc, char **argv);
int ffprobe_command_main(int argc, char **argv);
void ffmpeg_show_help_default(const char *opt, const char *arg);
void ffprobe_show_help_default(const char *opt, const char *arg);
void ffmpeg_options_reset(void);
void ffmpeg_common_reset(void);
void ffmpeg_common_uninit(void);
void ffmpeg_graph_reset(void);
void ffmpeg_log_reset(void);
int ffmpeg_host_descriptor(int fd);
ssize_t ffmpeg_host_read(int fd, void *buffer, size_t size);
ssize_t ffmpeg_host_write(int fd, const void *buffer, size_t size);
int ffmpeg_host_open(const char *path, int flags, ...);
FILE *ffmpeg_host_fopen(const char *path, const char *mode);
int ffmpeg_host_stat(const char *path, struct stat *st);
int ffmpeg_host_lstat(const char *path, struct stat *st);
int ffmpeg_host_access(const char *path, int mode);
int ffmpeg_host_unlink(const char *path);
int ffmpeg_host_rename(const char *from, const char *to);
int ffmpeg_host_mkdir(const char *path, mode_t mode);
int ffmpeg_host_rmdir(const char *path);
DIR *ffmpeg_host_opendir(const char *path);
int ffmpeg_host_glob(const char *pattern, int flags,
    int (*errfunc)(const char *, int), glob_t *result);

// Included only in this private FFmpeg build, after the system declarations.
// Worker threads share the one active invocation; no ios_system TLS is accessed.
#ifdef FFMPEG_HOST_REDIRECT
#undef stdin
#undef stdout
#undef stderr
#define stdin ffmpeg_host_stream(0)
#define stdout ffmpeg_host_stream(1)
#define stderr ffmpeg_host_stream(2)
#define printf(...) fprintf(stdout, __VA_ARGS__)
#define vprintf(format, args) vfprintf(stdout, format, args)
#define puts(s) (fputs(s, stdout) < 0 ? EOF : fputc('\n', stdout))
#define putchar(c) fputc(c, stdout)
#define getchar() fgetc(stdin)
#define perror ffmpeg_host_perror
#define getenv ffmpeg_host_getenv
#define open ffmpeg_host_open
#define fopen ffmpeg_host_fopen
#define stat(path, st) ffmpeg_host_stat(path, st)
#define lstat(path, st) ffmpeg_host_lstat(path, st)
#define access ffmpeg_host_access
#define unlink ffmpeg_host_unlink
#define rename ffmpeg_host_rename
#define mkdir ffmpeg_host_mkdir
#define rmdir ffmpeg_host_rmdir
#define opendir ffmpeg_host_opendir
#define glob ffmpeg_host_glob
#endif

#endif
