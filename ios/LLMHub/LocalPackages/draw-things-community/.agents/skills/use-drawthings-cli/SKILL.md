---
name: use-drawthings-cli
description: Use and troubleshoot an already installed Homebrew draw-things-cli for model discovery, authentication, local, cloud, or remote image generation, basic image-to-image and video generation, output verification, and safe agent operation. Use when a new user or agent needs to inspect Draw Things CLI commands or models, generate a first image, sign in to Draw Things cloud, connect to a Draw Things server, or diagnose common generation failures after installation.
---

# Use Draw Things CLI

Use the Homebrew-installed `draw-things-cli` executable from `PATH`. Assume it is already installed
and the user does not have the Draw Things source repository. Do not build with Bazel or SwiftPM
unless the user explicitly asks to develop or test the CLI itself.

## Follow The Shortest Safe Workflow

1. Verify the installed command and read its current help; repair installation only if verification
   fails.
2. Choose local, cloud, or remote-server execution.
3. Discover a valid model file ID instead of guessing one.
4. Start with one inexpensive, reproducible output.
5. Write to an explicit output path and validate the artifact before reporting success.

Before downloading a large local model or consuming cloud compute, tell the user what will happen.
Treat an explicit request to download or generate as authorization for that requested operation; do
not submit extra variants without permission. Cloud generation has no dry-run or built-in price
preview, so obtain confirmation immediately before the first cloud job unless the user explicitly
requested that cloud job.

## Verify The Installed CLI

Start here; do not reinstall or upgrade a working CLI merely to run a generation:

```sh
command -v draw-things-cli
draw-things-cli --version
draw-things-cli --help
draw-things-cli generate --help
```

If the executable is missing, confirm the stable Homebrew Core formula and install it as a recovery
step:

```sh
brew info draw-things-cli
brew install draw-things-cli
command -v draw-things-cli
```

If the installed CLI lacks a required documented option, record its version and ask before
upgrading:

```sh
brew outdated draw-things-cli
brew upgrade draw-things-cli
```

Do not substitute the `draw-things` GUI cask, an old custom tap, or `--HEAD`. On Apple Silicon, if
installation fails or the binary has the wrong architecture, compare `uname -m` with
`brew --prefix`; a native Homebrew normally uses `/opt/homebrew`, while `/usr/local` may indicate a
Rosetta installation. Do not replace Homebrew or edit shell startup files without the user's
approval. Use `brew info draw-things-cli` as the authority for current platform requirements.

Treat the installed `--help` output as the source of truth because flags and model support evolve.
Read the relevant subcommand help before constructing a non-trivial command:

```sh
draw-things-cli models --help
draw-things-cli auth --help
draw-things-cli generate --help
draw-things-cli train --help
```

## Choose A Backend

| Backend | Selection | Preparation | Current output boundary |
| --- | --- | --- | --- |
| Local Mac | default | Download model files locally | PNG and supported local video formats |
| Draw Things cloud | `--cloud-compute` | Run `auth login`; may consume paid compute | PNG only |
| Draw Things server | `--remote` | Obtain host, TLS, port, and optional secret | PNG only |

Keep these current CLI boundaries explicit:

- Remote and cloud generation reject video output.
- Remote and cloud generation reject `--audio`.
- LongCat `--avc` is local-only.
- Remote and cloud accept a normal `--image` input for supported image-to-image generation.
- `--offline` cannot be combined with remote or cloud execution.
- `--remote` and `--cloud-compute` are mutually exclusive.
- Remote and cloud upload prompts and any image input to the selected service. Confirm that the
  user accepts this privacy boundary.
- Remote and cloud do not upload local model or LoRA weights. Do not assume an unregistered local
  model or LoRA exists on the target service.

If the installed help differs, follow the installed version and report the difference.

## Discover And Prepare Models

List model mappings before choosing a model:

```sh
draw-things-cli models list
draw-things-cli models list --downloaded-only
```

Prefer the exact file ID from the first column, such as
`z_image_turbo_1.0_q8p.ckpt`. Human-readable names and Hugging Face references are supported, but
the exact registered file ID makes runs easier to reproduce.

For local generation, download the selected model and its registered dependencies explicitly when
the user wants a predictable preparation step:

```sh
draw-things-cli models ensure --model z_image_turbo_1.0_q8p.ckpt
```

Generation downloads missing files by default. Add `--no-download-missing` only after verifying the
files exist. Add `--offline` only when the user requires no network access.

Let the CLI resolve its default model directory unless the user supplies one. Resolution is shown
by `models list --help`; override it consistently with either `--models-dir /path/to/Models` or
`DRAWTHINGS_MODELS_DIR`. Quote paths containing spaces.

## Generate A First Local Image

Use model-recommended settings for the first run. Override only the seed and output so the result is
reproducible and saved:

```sh
draw-things-cli generate \
  --model z_image_turbo_1.0_q8p.ckpt \
  --prompt "A small red cube on a wooden table, soft studio light" \
  --seed 42 \
  --disable-preview \
  --output "$PWD/draw-things-first.png"
```

If that model ID is absent from `models list`, select another downloaded or official text-to-image
model and keep its recommended settings. Do not assume one model's steps or CFG suit another model.

Always pass `--output` for unattended agent work. Without it, the CLI may render a temporary inline
preview and intentionally leave no file.

## Generate With Draw Things Cloud

Prefer browser login over asking the user to paste an API key. Pause for the user when the browser,
Google login, MFA, or an OS permission prompt requires interaction:

```sh
draw-things-cli auth login
draw-things-cli auth token
```

Do not print saved credentials or place an API key in shell history, logs, source files, or chat.
Before submission, choose a new `.png` output path and confirm its parent directory exists and is
writable. This avoids paying for a successful generation that cannot be saved locally. After the
user has authorized cloud usage, generate one PNG first:

```sh
draw-things-cli generate \
  --cloud-compute \
  --model z_image_turbo_1.0_q8p.ckpt \
  --prompt "A small red cube on a wooden table, soft studio light" \
  --seed 42 \
  --output "$PWD/draw-things-cloud.png"
```

Run `auth logout` only when the user asks to remove saved credentials.

## Generate On A Draw Things Server

Ask for the host and whether it uses TLS. TLS is enabled by default. Disable it only for a trusted
local endpoint that the user identifies as non-TLS. Choose a new writable `.png` output path before
connecting:

```sh
draw-things-cli generate \
  --remote \
  --remote-url 127.0.0.1 \
  --remote-port 7859 \
  --no-remote-tls \
  --model z_image_turbo_1.0_q8p.ckpt \
  --prompt "A small red cube on a wooden table, soft studio light" \
  --seed 42 \
  --output "$PWD/draw-things-remote.png"
```

Do not expose `--remote-shared-secret` in logs or chat. If a secret is required and no protected
input mechanism is available, ask the user to run that authenticated command themselves.

## Use Common Input Modes

For local image-to-image generation, choose a model that supports it and use a strength in
`0...1`:

```sh
draw-things-cli generate \
  --model z_image_turbo_1.0_q8p.ckpt \
  --prompt "Transform this into a cinematic night scene" \
  --image /path/to/input.png \
  --strength 0.35 \
  --seed 42 \
  --output /path/to/img2img.png
```

Use `--prompt-file` for long or multiline prompts:

```sh
draw-things-cli generate \
  --model z_image_turbo_1.0_q8p.ckpt \
  --prompt-file /path/to/prompt.txt \
  --seed 42 \
  --output /path/to/result.png
```

For local video, first choose a video-capable model from `models list`, then consult
`generate --help` and the model's recommended settings. A basic command shape is:

```sh
draw-things-cli generate \
  --model ltx_2.3_22b_distilled_q6p.ckpt \
  --prompt "Ocean waves at sunset" \
  --frames 49 \
  --seed 42 \
  --video-format h264 \
  --output /path/to/clip.mp4
```

Respect model-specific frame-count and dimension constraints. Width and height overrides must be
multiples of 64.

## Preserve Configuration Semantics

Apply configuration in this order:

1. Begin with model-recommended settings.
2. Merge `--config-json` or `--config-file` only when needed.
3. Apply explicit flags such as `--steps`, `--cfg`, `--width`, `--height`, `--frames`, and `--seed`.
4. Use `--negative-prompt` only when intentionally replacing the recommended negative prompt.

Keep the full command, CLI version, model ID, seed, and output path when reproducibility or
benchmarking matters. Change one variable at a time during comparisons.

## Validate Every Result

Require a nonempty file before reporting success:

```sh
test -s /path/to/result.png
file /path/to/result.png
sips -g pixelWidth -g pixelHeight -g format /path/to/result.png
```

Also require exit status `0` and the CLI's `Wrote:` message. If a command returns multiple images,
the CLI may write numbered paths such as `result-0000.png`; validate every reported path instead of
assuming the original basename exists. Open or inspect the image with the agent's available image
viewer. For video, verify the container,
dimensions, duration, frame count, video codec, and audio stream with `ffprobe` when available, or
with macOS `mdls` and AVFoundation tools. Include the output path and measured generation timing in
the handoff.

Use a new output path by default. Do not overwrite an existing image or video unless the user has
explicitly authorized replacement; local video export can remove an existing destination before
writing the new container.

## Troubleshoot Deliberately

- `draw-things-cli: command not found`: confirm Homebrew works, run `eval "$(brew shellenv)"`, then
  retry `command -v draw-things-cli`.
- Unknown model: run `models list` and use an exact file ID from its output.
- Missing model or dependency: run `models ensure --model FILE_ID` without
  `--no-include-dependencies`.
- No output file: rerun with an explicit `--output` path.
- Cloud authentication failure: run `auth token`; if validation fails, run interactive
  `auth login` again.
- Remote connection failure: verify host, port, TLS choice, server availability, and shared-secret
  requirements before retrying.
- Remote/cloud video or audio error: use local generation; the current remote protocol does not
  carry those outputs or audio conditioning.
- Invalid dimensions: use width and height values divisible by 64.
- A long quiet local run: distinguish model download, graph compilation, sampling, and export; do
  not launch duplicate generation processes merely because output pauses.
- Suspected version mismatch: record `--version`, inspect current help, then use Homebrew to upgrade
  if the user wants the newer behavior.

## Hand Off Advanced Workflows

- Use `$run-longcat-avatar-video` for LongCat reference-image and audio video, AVC 93/13, checkpoint
  selection, performance comparison, and MP4 validation.
- Use `$train-lora` for LoRA training validation, loss and scaler checks, checkpoint inspection, and
  base-versus-LoRA comparisons.
- Use `$init-gpu-server` for provisioning a remote CUDA server; do not fold SSH, Docker, drivers, or
  disk setup into this onboarding workflow.

When those skills are unavailable, read the relevant installed CLI help and proceed conservatively
instead of inventing flags.
