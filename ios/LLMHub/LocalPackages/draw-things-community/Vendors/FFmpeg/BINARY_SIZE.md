# FFmpeg external library size measurements

Measured on 2026-09-27 using FFmpeg 9.0.2, iPhoneOS 27.0 SDK, ARM64, iOS 15.4
minimum deployment target, `-O2`, native ARM assembly/NEON, and no sanitizers.
Compiler: Apple clang 21.0.0 (`clang-2100.3.34.2`), Xcode 27.0. The intermediate
symbol-isolation link uses `ld-classic` from Xcode 26.6.

## Per-library increase

Each row compares **GPL built-ins plus only that library** against **GPL
built-ins alone**. The GPL baseline is shared and must not be added eight times.
All values are MiB (1,048,576 bytes).

| Library | Added capability | Linked increase | Gzip increase |
| --- | --- | ---: | ---: |
| SVT-AV1 4.2.0 | AV1 encoder | +3.116 | +1.414 |
| dav1d 1.5.4 | AV1 decoder | +0.690 | +0.328 |
| Opus 1.6.1 | Opus encoder/decoder | +0.344 | +0.214 |
| LAME 4.0 | MP3 encoder | +0.188 | +0.104 |
| libvmaf 3.2.1 | VMAF filter and embedded models | +0.898 | +0.239 |
| x264 r3222 | H.264 encoder, 8/10 bit | +1.191 | +0.587 |
| x265 4.3 | HEVC encoder, 8/10/12 bit | +8.141 | +2.870 |
| libvpx 1.17.0 | VP8/VP9 encoder/decoder, including high bit depth | +1.349 | +0.683 |

## Combined result

| Configuration | Stripped linked size | Gzip size |
| --- | ---: | ---: |
| Previous built-in-only LGPL configuration | 20.078 | 9.448 |
| Built-ins with GPL enabled | 20.391 | 9.586 |
| GPL plus all eight libraries | **36.322** | **16.014** |

Enabling GPL alone adds **0.313 MiB** linked
(0.138 MiB gzip). All eight libraries add
**15.931 MiB** beyond that GPL baseline.
The complete change from the previous LGPL configuration adds
**16.244 MiB linked**, or
**6.566 MiB gzip**.

The combined result is measured directly; it is not the sum of rounded rows.
Shared code, archive extraction, alignment, and compression make these figures
slightly non-additive. x265 is the largest contributor because this build
includes its separate 8-, 10-, and 12-bit implementations, matching Homebrew's
multilib configuration. VMAF includes the upstream default embedded models.

## What is measured

`SizeProbe.c` retains both production entry points (`FFmpegCommandRun` and
`FFprobeCommandRun`). Each configuration links the identical fixture with
`-dead_strip`, strips local/debug symbols, and removes the code signature.
The production archive hides every implementation symbol except those two entry
points. FFmpeg/FFprobe share the same codec code; their sizes are not added twice.

These are **incremental linked-code measurements**, not complete app sizes or
App Store download sizes. The fixture excludes other app code, assets, signing,
and license text. Gzip level 9 is a reproducible compression proxy; App Store
thinning, encryption, and compression differ. Source paths embedded by upstream
configuration strings can cause small compressed-size differences between runs.

[BINARY_SIZE.json](BINARY_SIZE.json) preserves the exact byte counts, including
the static archive size for every configuration. Static archive size is useful
for build storage but is not used as the shipped-code estimate. Source versions,
URLs, and SHA-256 pins are in [codec_sources.bzl](codec_sources.bzl).

## Reproduce

Host prerequisites: Xcode, CMake, and Python 3.10+. Bazel builds pinned Ninja and
pkgconf sources and runs pinned Meson sources. No Homebrew codec binaries are linked.
The reported build and runtime validation use ARM64; Intel builds have not been
validated and require an x86 assembler in addition to these host tools.

```sh
bazel build //Vendors/FFmpeg:size_report \
  --ios_multi_cpus=arm64 --cpu=ios_arm64 --apple_platform_type=ios \
  --ios_minimum_os=15.4 --jobs=3
```

For Xcode without `ld-classic`, also supply
`--action_env=FFMPEG_LD_CLASSIC=/path/to/ld-classic`.
The outputs are `bazel-bin/Vendors/FFmpeg/size_*.json` and the corresponding
`.macho` files. `size_baseline` is LGPL with no external libraries; `size_gpl` is
GPL with no external libraries; `size_all` is the production configuration.
Each other `size_<library>` target includes only the named external library.

```sh
bazel test //Vendors/FFmpeg:FFmpegCommandTests \
  //Vendors/FFmpeg:IOSSystemFFmpegTests --macos_minimum_os=13.5
bazel test //Vendors/FFmpeg:FFmpegCommandTests --config=asan --macos_minimum_os=13.5
bazel build //Apps/LocalCode:LocalCode --ios_multi_cpus=arm64
```

Runtime tests encode/decode H.264, HEVC, VP8, VP9, AV1, Opus, and MP3; explicitly
exercise dav1d; encode 10/12-bit HEVC; score identical and distorted frames with
VMAF; and repeat successful and failed initialization under the shared execution
lock. They check frame/sample counts, file descriptors, threads, live allocations,
parser behavior, cancellation, and session streams/paths. Runtime testing is on
macOS ARM64; iOS validation is a cross-build, not a physical-device execution.

The final validation passed both command suites and the full iOS app build.
AddressSanitizer reported **0 bytes of live-allocation growth after warmup**;
file-descriptor and thread counts remained unchanged. The ASan test now requires
zero growth, catching the x265 invalid-preset leak that the native allocator's
looser accounting threshold initially allowed. The x265 NEON filter patch was
also checked against a scalar reference on 1,000 random inputs per block size
(4, 8, 16, 32) and bit depth (8, 10, 12), with identical results.
