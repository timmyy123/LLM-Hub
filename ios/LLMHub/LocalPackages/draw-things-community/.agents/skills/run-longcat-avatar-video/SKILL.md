---
name: run-longcat-avatar-video
description: Generate, benchmark, validate, and troubleshoot LongCat-Video-Avatar 1.5 videos with draw-things-cli. Use when Codex needs to prepare LongCat q8p or i8x checkpoints, drive an avatar from a reference image and audio file, generate a fixed 4k+1 frame clip, run local segmented AVC for long audio with the canonical 93/13 configuration, compare i8x against q8p on Apple silicon, estimate full-video runtime, or verify LongCat MP4 frame count, duration, codec, and audio muxing.
---

# Run LongCat Avatar Video

Use the local `draw-things-cli generate` command for LongCat-Video-Avatar 1.5. Keep i8x and
q8p comparisons identical except for the model checkpoint and output path.

## Build And Inspect The CLI

Work from the Draw Things repository root. Build once, then reuse the same optimized binary for
all runs in one comparison:

```sh
bazel build --compilation_mode=opt //Apps:DrawThingsCLI
CLI=bazel-bin/Apps/DrawThingsCLI
"$CLI" generate --help
```

Prefer the app model directory on macOS unless the user provides another one:

```sh
MODELS_DIR="${DRAWTHINGS_MODELS_DIR:-$HOME/Library/Containers/com.liuliu.draw-things/Data/Documents/Models}"
```

Do not switch builds, power modes, resolutions, segment sizes, or preview settings during a
benchmark.

## Prepare Models

Choose one of these DiT checkpoints:

- `longcat_video_avatar_1.5_dmd_i8x.ckpt`: 8-bit S model; prefer it for throughput on supported
  Apple silicon.
- `longcat_video_avatar_1.5_dmd_q8p.ckpt`: q8p baseline and fallback; use it for matched quality
  and performance comparisons.

Ensure the selected model and its registered dependencies:

```sh
"$CLI" models ensure \
  --models-dir "$MODELS_DIR" \
  --model longcat_video_avatar_1.5_dmd_i8x.ckpt
```

LongCat also needs these files in `MODELS_DIR`:

```text
umt5_xxl_encoder_q8p.ckpt
wan_v2.1_video_vae_f16.ckpt
whisper_large_v3_f16.ckpt
```

The current model dependency list covers UMT5 and the Wan VAE. Verify Whisper separately because
`--audio-encoder-file` defaults to `whisper_large_v3_f16.ckpt`, but it is not a registered LongCat
model dependency:

```sh
test -f "$MODELS_DIR/whisper_large_v3_f16.ckpt"
```

Pass `--audio-encoder-file NAME.ckpt` only when using a different compatible Whisper checkpoint.

## Audio API Boundary

Keep model audio separate from ControlNet hints. The reusable input and model-specific encoders live
in `Libraries/AudioConverter`; `LocalImageGenerator` only consumes finished conditioning. LongCat
audio follows this internal path:

```text
AudioInput -> LongCatAudioConditioningEncoder -> LongCatAudioFeatures
           -> LongCatAudioConditioning -> AudioConditioning.longCat
```

`AudioInput` owns decoded PCM and builds the waveform used for output audio muxing. Run the Whisper
encoder once to produce `LongCatAudioFeatures`; derive one `LongCatAudioConditioning` for a normal
generation or one per AVC segment. `AudioConditioning` is the model-dispatch boundary where a future
LTX audio-conditioning case can be added. Do not represent model audio as `ControlHintType.audio` or
route it through `ControlModel`.

## Validate Inputs

Use both a reference image and driving audio. The image is aspect-scaled and center-cropped to the
requested output size. Width and height must be multiples of 64.

Confirm that AVFoundation can decode the audio before starting a long run:

```sh
afinfo "$AUDIO"
```

Reject an input that reports zero packets or zero duration even if its file size is nonzero. Use a
valid CAF, MP3, M4A, or WAV container that `afinfo` and `AVAudioFile` can decode; do not fix a bad
container by renaming its extension.

The DMD checkpoint is step-distilled. Use these baseline values unless the task explicitly changes
them:

```text
steps = 8
cfg = 1
shift = 7
fps = 25
```

LongCat temporal counts must be `4k + 1`. Canonical values are 93 generated frames and 13 AVC
condition frames. Values such as 77 and 109 are valid experiments, but do not mix them into a
canonical 93/13 comparison.

## Generate An AVC Video

Use AVC for audio longer than one generated clip. AVC is local-only and currently supports only
LongCat-Video-Avatar 1.5. Its output duration follows the audio; do not pass `--frames` with
`--avc`.

Use 93/13 explicitly even though they are the current defaults:

```sh
IMAGE=/path/to/reference.png
AUDIO=/path/to/driving-audio.caf
OUTPUT=/path/to/longcat_avc_i8x.mp4
PROMPT='A person speaks naturally to the camera with stable posture and synchronized mouth motion.'

"$CLI" generate --avc \
  --models-dir "$MODELS_DIR" \
  --model longcat_video_avatar_1.5_dmd_i8x.ckpt \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "$PROMPT" \
  --steps 8 --cfg 1 \
  --segment-frames 93 --cond-frames 13 \
  --width 448 --height 320 \
  --seed 42 \
  --config-json '{"shift":7}' \
  --no-download-missing \
  --disable-preview \
  --video-format h264 \
  --output "$OUTPUT"
```

With 93/13, each later segment contributes 80 new frames. For an audio target of `T` frames:

```text
stride = 93 - 13 = 80
segments = T <= 93 ? 1 : ceil((T - 93) / 80) + 1
sampling steps = segments * 8
```

The CLI computes Whisper features for the generated span, reuses the last 13 decoded frames as the
next segment's clean condition, drops overlap frames, trims to the audio target, and muxes the input
speech into the output container.

For the full approximately 82-second reference workload, keep 93/13 and change only the audio,
resolution, and output:

```sh
"$CLI" generate --avc \
  --models-dir "$MODELS_DIR" \
  --model longcat_video_avatar_1.5_dmd_i8x.ckpt \
  --image "$IMAGE" --audio /path/to/man.mp3 \
  --prompt "$PROMPT" \
  --steps 8 --cfg 1 \
  --segment-frames 93 --cond-frames 13 \
  --width 832 --height 512 \
  --seed 42 --config-json '{"shift":7}' \
  --no-download-missing --disable-preview \
  --video-format h264 \
  --output /path/to/man_832x512_i8x.mp4
```

## Generate One 93-Frame Clip Without AVC

Omit `--avc`, `--segment-frames`, and `--cond-frames`. Set `--frames 93` explicitly:

```sh
"$CLI" generate \
  --models-dir "$MODELS_DIR" \
  --model longcat_video_avatar_1.5_dmd_i8x.ckpt \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "$PROMPT" \
  --steps 8 --cfg 1 \
  --frames 93 \
  --width 448 --height 320 \
  --seed 42 \
  --config-json '{"shift":7}' \
  --no-download-missing \
  --disable-preview \
  --video-format h264 \
  --output /path/to/longcat_93f_i8x.mp4
```

At 25 fps, 93 frames produce 3.72 seconds. The exported audio is trimmed or padded to the same
duration.

## Compare i8x And q8p

Run comparisons serially on the same machine. Keep all of these identical:

- CLI binary and build mode
- image, audio, and prompt
- width, height, and frame or AVC settings
- steps, CFG, shift, and seed
- preview and video-format flags
- system power mode and competing workload

Change only:

```text
--model longcat_video_avatar_1.5_dmd_i8x.ckpt
--model longcat_video_avatar_1.5_dmd_q8p.ckpt
```

Wrap each command with `/usr/bin/time -p`. Record both the CLI's generation/sampling summary and
the external `real` time:

```sh
/usr/bin/time -p "$CLI" generate ...
```

Do not call model-loading, sampling-step, generation, and wall-clock speedups interchangeable.
Report each metric by name.

### Measured M5 Max Baseline

The following results were measured on 2026-07-13 on an Apple M5 Max with 48 GB memory. Both models
used `448x320`, 8 steps, CFG 1, shift 7, seed 42, H.264, disabled preview, and the same image, prompt,
and valid 7.988-second audio. Use these as reference data, not a universal performance guarantee.

| Workload | Model | Sampling steps | Generation | Median / step | Wall time |
| --- | --- | ---: | ---: | ---: | ---: |
| AVC 93/13, 200 output frames | i8x | 24 | 368.45 s | 10.69 s | 421.77 s |
| AVC 93/13, 200 output frames | q8p | 24 | 497.04 s | 14.92 s | 550.07 s |
| No AVC, 93 output frames | i8x | 8 | 86.59 s | 7.98 s | 134.70 s |
| No AVC, 93 output frames | q8p | 8 | 136.40 s | 13.23 s | 184.26 s |

Observed speedups from that matched run:

- AVC wall time: i8x was 1.30x faster; median sampling step was 1.40x faster.
- Non-AVC wall time: i8x was 1.37x faster; median sampling step was 1.66x faster.

An earlier `832x512`, approximately 82-second, AVC 93/13 workload was roughly 9 hours with q8p and
3 hours or more with i8x. Treat that as a historical high-resolution observation, not a canonical
3x claim, because it was not captured with the same benchmark ledger as the table above. Re-run
both checkpoints with matched commands before publishing a 3x result.

## Verify Outputs

Check existence, size, codecs, dimensions, and duration:

```sh
ls -lh "$OUTPUT"
mdls \
  -name kMDItemCodecs \
  -name kMDItemDurationSeconds \
  -name kMDItemPixelWidth \
  -name kMDItemPixelHeight \
  "$OUTPUT"
```

On macOS, count actual video samples with AVFoundation when `ffprobe` is unavailable:

```sh
VIDEO="$OUTPUT" xcrun swift -e '
import AVFoundation
import Foundation
let asset = AVURLAsset(url: URL(fileURLWithPath: ProcessInfo.processInfo.environment["VIDEO"]!))
let track = asset.tracks(withMediaType: .video).first!
let reader = try AVAssetReader(asset: asset)
let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
reader.add(output)
reader.startReading()
var samples = 0
while let buffer = output.copyNextSampleBuffer() {
  samples += CMSampleBufferGetNumSamples(buffer)
}
print("video_samples=\(samples) status=\(reader.status.rawValue)")
'
```

Expect 93 samples for `--frames 93`. For AVC, expect the audio-derived target; a valid 7.988-second
input at 25 fps produces 200 samples and an 8.00-second MP4.

Inspect visual continuity around each AVC boundary. With 93/13, the first output boundary is near
frame 93, and later boundaries advance by 80 frames. Compare i8x and q8p boundary frames before
making a quality claim.

## Troubleshoot

- `--frames must be 4k + 1`: use 93 for the canonical single-shot run.
- `--segment-frames` or `--cond-frames` validation fails: use 93/13; both values must be `4k + 1`,
  and segment frames must exceed condition frames.
- `--frames cannot be used with --avc`: remove `--frames`; AVC duration follows audio.
- `--avc currently supports only local generation`: remove `--remote` and `--cloud-compute`.
- Missing Whisper error: place the selected audio encoder in `MODELS_DIR` or pass
  `--audio-encoder-file` with its filename.
- Audio reports zero duration or zero packets: replace or properly transcode the source file before
  generation.
- High-resolution run exhausts memory: validate first at `448x320`, close competing GPU workloads,
  then retry `832x512`. Do not change segment size during an i8x/q8p comparison.
- Unexpectedly weak speedup: compare sampling metrics separately from Whisper, model loading, VAE,
  and video encoding. Fixed costs dominate short low-resolution clips.
- Segment boundary jump: confirm both runs use the same 93/13, seed, audio, and reference image;
  inspect frames around each 80-frame stride before changing the continuation policy.
- `--zero-audio-features` is a hidden pipeline diagnostic. Never use it for a quality result.

Keep the current stable LongCat behavior unless debugging model internals. In particular, do not
change masked-reference attention or continuation policy merely to improve a benchmark number.
