// Only linked into upstream configure checks. Never included in the archive.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
FILE *ffmpeg_host_stream(int fd) { return fd == 0 ? stdin : fd == 1 ? stdout : stderr; }
FILE *ffmpeg_host_fopen(const char *path, const char *mode) { return fopen(path, mode); }
char *ffmpeg_host_getenv(const char *name) { return getenv(name); }
int ffmpeg_host_rename(const char *from, const char *to) { return rename(from, to); }
int ffmpeg_host_unlink(const char *path) { return unlink(path); }
