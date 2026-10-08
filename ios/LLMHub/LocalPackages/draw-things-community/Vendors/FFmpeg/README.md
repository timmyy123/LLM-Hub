# FFmpeg and FFprobe command integration

FFmpeg 9.0.2 is downloaded from the upstream release archive with a SHA-256 pin
in `repositories.bzl`. Bazel builds it from source for the selected Apple SDK,
architecture, and deployment target. No downloaded executable is launched.

`//Vendors/FFmpeg:FFmpeg` exposes the synchronous C functions `FFmpegCommandRun`
and `FFprobeCommandRun`, sharing one context type and one libav build.
`//Vendors/FFmpeg:command` registers conventional `ffmpeg_main(argc, argv)` and
`ffprobe_main(argc, argv)` adapters through the existing ios_system registry.
Local Code schedules both commands in the FFmpeg operation queue. A mutex inside
the library enforces serialization even for calls from OSH, Python, or other hosts.

## Command syntax and components

The command uses FFmpeg's original `ffmpeg_parse_options`, option tables, stream
specifiers, and filter parser. FFprobe uses its original `parse_options`, option
tables, `-show_entries`/stream selection handlers, and output formatters (including
JSON and CSV). The adapter does not parse or rewrite arguments.
The default GPL build also includes SVT-AV1, dav1d, Opus, LAME, libvmaf, x264,
x265, and libvpx, plus VideoToolbox, AudioToolbox, SecureTransport, and zlib.
Releases and SHA-256 checksums are pinned in `codec_sources.bzl`. Each library
uses its upstream build system and produces a static archive for the selected
SDK. Host CMake and Python 3.10+ must be on PATH; Ninja, Meson, and pkgconf sources
are pinned and build tools are isolated from target libraries. Native
ARM assembly and FFmpeg worker threads remain enabled. Device capture/output is
disabled, except the `lavfi` input device. `ffplay` is not built.
Use `ffmpeg -codecs`, `-filters`, `-formats`, and `-protocols` to inspect the
compiled capabilities. GPL components are enabled; nonfree components are not.

Examples:

```sh
ffmpeg -i input.mov -c:v h264_videotoolbox -c:a aac output.mp4
ffmpeg -i input.mp4 -vf scale=640:-2 -frames:v 1 preview.png
ffmpeg -i input.wav -ar 16000 -ac 1 -f s16le pipe:1 > audio.raw
ffprobe -v error -show_format -show_streams -of json input.mp4
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 input.mp4
```

## Invocation ownership

The execution lock covers option initialization, file opening, transcoding,
worker shutdown, and cleanup. The host context and its strings/streams are
borrowed until return. Worker threads access the same active invocation rather
than ios_system's command-thread TLS. The adapter snapshots the environment and
directory aliases; relative filesystem paths use the session directory.

`ffmpeg-ios.patch` resets mutable frontend options and statistics, clears closed
handles, closes progress/report streams on error paths, and releases embedded
resources. It preserves upstream parsing and media execution. The archive binds
and hides all implementation symbols except `FFmpegCommandRun` and
`FFprobeCommandRun`, so libav logging, CPU overrides, and allocation settings
cannot affect another libav consumer.

FFprobe resets its section selections, format options, counters, and logging
state between calls. Decoder setup errors return through normal cleanup instead
of exiting the process. Failed formatter initialization closes the already-open
writer. Shared help dispatch and program metadata select the active frontend
under the same execution lock.

Cancellation is cooperative through the host token, the existing FFmpeg I/O
interrupt callback, and scheduler shutdown. Cancellation while queued does not
enter either frontend. FFprobe uses the host cancellation token in its input I/O
interrupt callback and packet loop. Pipe reads/writes poll for cancellation.
Neither frontend installs process signal handlers or alters terminal modes; keyboard interaction is
disabled, while media on stdin remains supported. `-timelimit` returns an
unsupported error rather than changing the application's CPU resource limit.

The current OSH shell buffers pipeline stages with its existing 16 MiB limit.
Sequential FFmpeg/FFprobe stages therefore work within that limit; this is not an
unbounded streaming shell. Hosts must not connect simultaneous FFmpeg/FFprobe
invocations with a live bounded pipe: serialization would prevent both stages
from making progress.

## External codec ownership

External libraries share the same command lock, private symbols, and session
file/environment/stream hooks. SVT-AV1 receives a permanent context-free logging
callback that resolves the active invocation; its upstream default logger would
otherwise retain the first command's FILE pointer. `SVT_LOG` is read per callback;
`SVT_LOG_FILE` is not used (redirect command stderr instead).

x264 supports 8/10-bit input and x265 includes 8/10/12-bit implementations. libvpx
includes VP8, VP9, and high-bit-depth VP9. dav1d includes both bit-depth variants.
VMAF embeds its upstream default models so scoring does not need a downloaded
model file. ARM NEON remains enabled; SVE is disabled for Apple targets.
`x265-neon.patch` fixes an upstream first-vector underread in the NEON reference
filter for all three bit depths; it preserves the filter output and is covered
by the sanitizer encode tests. The FFmpeg patch also enables cleanup after any
libx265 initialization failure (including invalid presets), with an idempotent
close path. Upstream only closes some failed initialization paths.

See [BINARY_SIZE.md](BINARY_SIZE.md) for the per-library linked-size comparison
and reproduction commands. Size fixtures are manual targets and are not shipped with the app.

See [TESTING.md](TESTING.md) for executable coverage, the sanitizer soak command,
and prioritized device, I/O-fault, corpus, and lifecycle tests.

## Validation and upgrades

With Xcode versions that omit `ld-classic`, pass
`--action_env=FFMPEG_LD_CLASSIC=/path/to/ld-classic` to Bazel to use an installed
linker for the relocatable symbol-isolation step.

```sh
bazel test //Vendors/FFmpeg:FFmpegCommandTests //Vendors/FFmpeg:IOSSystemFFmpegTests --macos_minimum_os=13.5
bazel test //Vendors/FFmpeg:FFmpegCommandTests --config=asan --macos_minimum_os=13.5
bazel build //Apps/LocalCode:LocalCode --ios_multi_cpus=arm64
```

The tests exercise the actual upstream parser, audio/video sample counts,
repeated success and partial initialization failure, option reset, descriptor
and live-allocation growth, serialized admission, cancellation, signal
dispositions, FFprobe JSON metadata and frame counts, and real ios_system/OSH
streams, environment and paths. External codec tests also cover encode/decode
round trips, 10/12-bit HEVC, VMAF scoring, and failed encoder/filter initialization.

When upgrading, update the release checksum and rebase the patch. Re-audit
file-scope and function-static mutable state in fftools, including shared
command utilities, logging, resources and newly introduced frontend modules.
Retain the same upstream parser and run both test suites and the iOS app build.

## Notices

Upstream: https://ffmpeg.org/
Source: https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz
SHA-256: `8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e`

FFmpeg is copyright its respective contributors. The default build uses the
GPL 2.0 or later configuration because x264 and x265 are enabled. See
`COPYING.GPLv2`, `COPYING.LGPLv2.1`, upstream `LICENSE.md`, and `Licenses/` for
library license and patent notices. This software is based in part on the
work of the Independent JPEG Group. The three IJG-derived files identified in
`LICENSE.md` are unmodified. Local modifications are recorded in
`ffmpeg-ios.patch`, `x265-neon.patch`, and the host adapter sources alongside them.
