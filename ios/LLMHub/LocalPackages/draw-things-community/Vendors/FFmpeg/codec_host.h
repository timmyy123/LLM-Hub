#ifndef FFMPEG_CODEC_HOST_H
#define FFMPEG_CODEC_HOST_H
#ifndef __ASSEMBLER__
#ifdef __cplusplus
#include <cstdio>
#include <cstdlib>
extern "C" {
#else
#include <stdio.h>
#include <stdlib.h>
#endif
FILE *ffmpeg_host_stream(int fd);
FILE *ffmpeg_host_fopen(const char *path, const char *mode);
char *ffmpeg_host_getenv(const char *name);
int ffmpeg_host_rename(const char *from, const char *to);
int ffmpeg_host_unlink(const char *path);
#ifdef __cplusplus
}
namespace std {
using ::ffmpeg_host_fopen;
using ::ffmpeg_host_getenv;
using ::ffmpeg_host_rename;
}
#endif
#undef stdin
#undef stdout
#undef stderr
#define stdin ffmpeg_host_stream(0)
#define stdout ffmpeg_host_stream(1)
#define stderr ffmpeg_host_stream(2)
#define printf(...) fprintf(stdout, __VA_ARGS__)
#define vprintf(format, args) vfprintf(stdout, format, args)
#define puts(s) (fputs(s, stdout) < 0 ? EOF : fputc('\n', stdout))
#define fopen ffmpeg_host_fopen
#define getenv ffmpeg_host_getenv
#define rename ffmpeg_host_rename
#define unlink ffmpeg_host_unlink
#endif
#endif
