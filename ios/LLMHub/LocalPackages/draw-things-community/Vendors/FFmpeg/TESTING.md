# Embedded FFmpeg validation

The default configuration includes all eight external libraries and GPL components.
The central test requirement is that **one process survives a sequence of commands**:
a fresh CLI subprocess for each case cannot detect retained globals, stale streams,
or cleanup failures in this integration.

## Automated coverage

| Scenario | Assertion |
| --- | --- |
| Encode/decode all eight libraries | Exact decoded frame/sample sizes; explicit dav1d decode; 10/12-bit HEVC |
| Lossless x264, x265, VP9 | Decoded YUV bytes match a separately decoded FFV1 reference exactly |
| VMAF | Identical frames score highly; deliberately distorted frames score substantially lower |
| Partial initialization failures | Invalid x264/x265 presets and missing VMAF models fail; subsequent commands succeed |
| Damaged inputs | Empty and progressively truncated Matroska headers fail in both ffmpeg and ffprobe; valid input still works afterward |
| Two sessions | Interleaved x264 passes use identical relative statistics/output filenames in separate directories containing spaces; ffprobe sees each session's expected resolution; process CWD is unchanged |
| SVT logging lifetime | Fresh streams per invocation, alternating verbose/silent/verbose environments; diagnostics follow the current invocation |
| Active codec cancellation | Cancel x264, x265, VP8, VP9, and SVT after the progress report proves encoded frames exist; return 130 within the test deadline; a queued ffprobe then succeeds |
| Admission cancellation | A queued ffprobe can cancel without disturbing the running ffmpeg |
| Pipe failures | Blocked input cancellation and closed output reader; no process-wide SIGPIPE or signal-handler mutation |
| Actual shell adapter | OSH workflow traverses x264/Opus → x265/LAME → VP9 → SVT → dav1d, checks raw frames and parsed ffprobe metadata, runs VMAF, and verifies quoted filenames/environment/redirection |
| Repeated mixed commands | Stable file-descriptor/thread counts and zero live-allocation growth after warmup under ASan |

The native allocator's accounting has a 1 MiB tolerance because its retained
storage varies. ASan uses `__sanitizer_get_current_allocated_bytes()` and has
**no positive-growth allowance**. RSS alone is not a leak oracle: allocator
quarantine and reusable arenas can retain resident pages after objects are freed.

```sh
bazel test //Vendors/FFmpeg:FFmpegCommandTests \
  //Vendors/FFmpeg:IOSSystemFFmpegTests --macos_minimum_os=13.5
bazel test //Vendors/FFmpeg:FFmpegCommandTests \
  --config=asan --macos_minimum_os=13.5
bazel test //Vendors/FFmpeg:FFmpegCommandTests \
  --config=asan --macos_minimum_os=13.5 --test_env=FFMPEG_TEST_SOAK=1
bazel build //Apps/LocalCode:LocalCode --ios_multi_cpus=arm64
```

The normal lifecycle loop has 40 iterations; `FFMPEG_TEST_SOAK=1` runs 200.
Every tenth iteration includes all codecs, lossless comparisons, malformed
inputs, session alternation, and failed initialization. Active encoder
cancellation runs before the measured loop and again during it. Input-pipe,
ffprobe, parser/reset, cancellation, signal, and allocation-limit cases also run.
The environment variable controls only the test executable; it is not a product
or FFmpeg command option.

## Next tests, prioritized

These are follow-up ideas, not claims of completed coverage.

| Priority | Test | Why it matters / pass criteria |
| --- | --- | --- |
| 1 | Physical iPhone/iPad, smallest supported RAM tier | Run 1080p/4K, long audio, 10/12-bit, multiple encoder presets/thread counts. Record peak resident/physical footprint, post-command live memory, time to cancel, thermal state and any jetsam termination. Desktop ASan does not establish a safe device memory budget. |
| 1 | Output backpressure and filesystem faults | Fill a pipe with a reader that stays open without consuming, cancel while the writer is blocked; close stdout/stderr at different phases; use a failing stream to inject ENOSPC/EIO. Require bounded completion, closed resources and a successful next command. Test x264/x265 two-pass sidecars and VMAF report output too. |
| 1 | Broader media correctness matrix | Odd/edge dimensions, 4:2:0/4:2:2/4:4:4, 8/10/12-bit, mono/stereo/multichannel, variable frame rate, rotation/color metadata, seeking, multiple streams and stream copy. Compare lossless hashes, timestamp continuity and codec metadata against upstream reference runs; use tolerances for lossy output. |
| 2 | Corrupt packet corpus and fuzzing | Go beyond truncated container headers: mutate compressed packets, extradata, seek tables and nested format lengths. Replay a bounded corpus through the embedded runner under sanitizers, with duration/output/memory limits and per-case deadlines. Keep crashes reproducible with pinned seeds/input files. |
| 2 | Deterministic allocation-failure injection | Fail selected allocations before and after encoder/decoder/filter initialization, including allocations inside external libraries. Require a nonzero result, no double-free/leak and successful subsequent execution. FFmpeg's `-max_alloc` only covers libav allocations. |
| 2 | Simultaneous independent ios_system sessions | While one session encodes, let another run ordinary shell commands and queue probes; cancel/close one session. Assert that other commands remain responsive and streams, aliases, environments and cancellation tokens remain isolated. The current native runner test covers serialized codec admission, not this full application scenario. |
| 2 | Background/foreground and interrupted I/O on device | Exercise app lifecycle transitions, interrupted input/network streams and cancellation while suspended/resumed. Verify cleanup and recovery within the app's actual execution policy. |
| 3 | Hardware/software interaction | Interleave software codecs with VideoToolbox/AudioToolbox, including decode → filter → encode pipelines. Exercise format negotiation, hardware-frame cleanup and cancellation on physical devices. |
| 3 | Soak across changing settings | Seeded sequences alternating codec, resolution, depth, preset, threading, valid/invalid options, directory and environment for thousands of commands. Sample live memory/FDs/threads by phase to distinguish bounded caches from linear growth. |
| 3 | Upgrade gates | Run the suite against every pinned-library/FFmpeg update; inspect exported symbols, licensing, minimum deployment versions and per-library sizes. Run relevant upstream codec/FATE cases through the embedded adapter where practical. |

Device measurements should choose limits from observed peak usage and cancellation
latency; the linked-size report is not a runtime-memory budget. Cancellation is
cooperative, so tests should measure actual exit latency at expensive encode and
I/O stages rather than assuming a request stops work immediately.
