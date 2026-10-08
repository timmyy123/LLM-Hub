# MediaGenerationKit And CLI

This file is the combined quick-start and contract guide for:

- the `MediaGenerationKit` Swift package
- the `media-generation-kit-cli` example client

Full documentation:

- https://drawthingsai.github.io/media-generation-kit

Keep examples, help text, and behavior aligned with:

- `Libraries/MediaGenerationKit`
- `Apps/MediaGenerationKitCLI/MediaGenerationKitCLI.swift`

If command or API behavior changes, update this file and the wrapper-repo copy that is synced into
`drawthingsai/media-generation-kit`.

## Swift Package

### Installation

In `Package.swift`:

```swift
import PackageDescription

let package = Package(
  name: "MyApp",
  dependencies: [
    .package(
      url: "https://github.com/drawthingsai/media-generation-kit.git",
      revision: "37e0b70092ad1a0c1d4c6b24f16e17b73f1b1fb3"
    )
  ],
  targets: [
    .executableTarget(
      name: "MyApp",
      dependencies: [
        .product(name: "MediaGenerationKit", package: "media-generation-kit")
      ]
    )
  ]
)
```

`media-generation-kit` currently pins `draw-things-community` by revision, so install this package
with a specific revision instead of a version-based requirement.

### Public Surface

The package is centered on:

- `MediaGenerationPipeline`
- `MediaGenerationPipeline.Preview`
- `MediaGenerationEnvironment`
- `LoRAImporter`
- `LoRAStore`
- `MediaGenerationKitError`
- `AppCheckConfiguration`

Core rules:

- `MediaGenerationPipeline.fromPretrained(...)` is async.
- Configuration lives on `pipeline.configuration`.
- `MediaGenerationEnvironment.default` owns process-wide defaults and model-management helpers.
- `LoRAImporter` is local conversion only.
- `LoRAStore` is Draw Things cloud storage only.

### Minimal Swift App

```swift
import Foundation
import MediaGenerationKit

@main
struct ExampleApp {
  static func main() async throws {
    try await MediaGenerationEnvironment.default.ensure(
      "flux_2_klein_4b_q8p.ckpt"
    )

    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      "flux_2_klein_4b_q8p.ckpt",
      backend: .local
    )

    pipeline.configuration.width = 1024
    pipeline.configuration.height = 1024
    pipeline.configuration.steps = 4

    let results = try await pipeline.generate(
      prompt: "a cat in studio lighting",
      negativePrompt: ""
    )

    try results[0].write(
      to: URL(fileURLWithPath: "/tmp/cat.png"),
      type: .png
    )
  }
}
```

### Remote Swift App

```swift
import Foundation
import MediaGenerationKit

@main
struct RemoteExampleApp {
  static func main() async throws {
    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      "flux_2_klein_4b_q8p.ckpt",
      backend: .remote(.init(host: "127.0.0.1", port: 7859))
    )

    pipeline.configuration.width = 1024
    pipeline.configuration.height = 1024
    pipeline.configuration.steps = 4

    let results = try await pipeline.generate(
      prompt: "a cat in studio lighting",
      negativePrompt: ""
    )

    try results[0].write(
      to: URL(fileURLWithPath: "/tmp/cat.png"),
      type: .png
    )
  }
}
```

### Cloud Compute Swift App

```swift
import Foundation
import MediaGenerationKit

@main
struct CloudComputeExampleApp {
  static func main() async throws {
    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      "flux_2_klein_4b_q8p.ckpt",
      backend: .cloudCompute(apiKey: "YOUR_API_KEY")
    )

    pipeline.configuration.width = 1024
    pipeline.configuration.height = 1024
    pipeline.configuration.steps = 4

    let results = try await pipeline.generate(
      prompt: "a cat in studio lighting",
      negativePrompt: ""
    )

    try results[0].write(
      to: URL(fileURLWithPath: "/tmp/cat.png"),
      type: .png
    )
  }
}
```

### Backend Shape

- Local:
  - `backend: .local`
  - `backend: .local(directory: "/path/to/Models")`
- Remote:
  - `backend: .remote(.init(host: "127.0.0.1", port: 7859))`
- Cloud compute:
  - `backend: .cloudCompute(apiKey: "YOUR_API_KEY")`

### Inputs And Results

Inputs are direct values:

- `CIImage`
- `UIImage`
- `MediaGenerationPipeline.data(_:)`
- `MediaGenerationPipeline.file(_:)`
- role wrappers such as `.mask()`, `.moodboard()`, `.depth()`

There is no standalone request/options/assets object.

Results are `MediaGenerationPipeline.Result` values and support:

- `write(to:type:)`
- `CIImage(result)`
- `UIImage(result)`

Generation progress callbacks can also carry a `MediaGenerationPipeline.Preview?`:

- `preview == nil` for non-preview states such as `.preparing`, `.uploading`, or `.completed`
- `preview != nil` for preview-bearing progress updates such as sampling
- `Preview` is a random-access collection whose subscript lazily decodes a `CGImage` on demand

### Environment Helpers

Important members on `MediaGenerationEnvironment.default`:

- `externalUrls`
- `maxTotalWeightsCacheSize`
- `ensure(...)`
- `resolveModel(...)`
- `suggestedModels(...)`
- `inspectModel(...)`
- `downloadableModels(...)`

Sync vs async catalog rules:

- Sync overloads are offline-only or cache-only.
- Async overloads are the network-capable path.
- If a sync overload is called with `offline: false` and it would need uncached remote catalog data, it throws `MediaGenerationKitError.asyncOperationRequired(...)`.
- `suggestedModels(..., offline: false)` is stricter: the sync variant throws immediately if remote catalog data is not already cached.

### LoRA Flows

Local conversion:

- `LoRAImporter(file:version:)`
- optional `inspect()`
- `import(to:scaleFactor:progressHandler:)`

Cloud storage:

- `LoRAStore(backend:)`
- `upload(_:file:)`
- `delete(_:)`
- `delete(keys:)`

## CLI

### Command Shape

- One `generate` command handles both text-to-image and image-to-image.
- Remote generation is enabled with `--remote`, not with a separate subcommand.
- Draw Things cloud generation is enabled with `--cloud-compute`, not with a separate subcommand.
- `--models-dir` is an optional flag, never a positional argument.
- Auth commands are standalone and never require a models directory.
- Saved cloud credentials are reused by `auth state`, `auth token`, `lora upload`, and `generate --cloud-compute`.
- Saved cloud credentials remember the cloud API base URL used at login time.

### Canonical Generation

Local generation:

```bash
swift run -c release media-generation-kit-cli generate \
  --models-dir /tmp \
  --prompt "a cat" \
  --model "flux_2_klein_4b_q8p.ckpt" \
  --width 1024 \
  --height 1024 \
  --num-inference-steps 4 \
  --output /tmp/cat.png
```

Image-to-image:

```bash
swift run -c release media-generation-kit-cli generate \
  --models-dir /tmp \
  --prompt "a studio portrait" \
  --model "flux_2_klein_4b_q8p.ckpt" \
  --width 1024 \
  --height 1024 \
  --num-inference-steps 4 \
  --image /tmp/input.png \
  --strength 0.35 \
  --output /tmp/portrait.png
```

Remote generation:

```bash
swift run -c release media-generation-kit-cli generate \
  --remote \
  --remote-url 127.0.0.1 \
  --remote-port 7859 \
  --remote-tls \
  --prompt "a cat" \
  --model "flux_2_klein_4b_q8p.ckpt" \
  --width 1024 \
  --height 1024 \
  --num-inference-steps 4
```

Draw Things cloud compute:

```bash
swift run -c release media-generation-kit-cli generate \
  --cloud-compute \
  --api-key "API_KEY" \
  --prompt "a cat" \
  --model "flux_2_klein_4b_q8p.ckpt" \
  --width 1024 \
  --height 1024 \
  --num-inference-steps 4 \
  --output /tmp/cat.png
```

Saved-login cloud compute:

```bash
swift run -c release media-generation-kit-cli auth login

swift run -c release media-generation-kit-cli generate \
  --cloud-compute \
  --prompt "a cat" \
  --model "flux_2_klein_4b_q8p.ckpt" \
  --width 1024 \
  --height 1024 \
  --num-inference-steps 4 \
  --output /tmp/cat.png
```

### Auth Commands

Google browser sign-in:

```bash
swift run -c release media-generation-kit-cli auth login
```

Google browser sign-in against a non-default API host:

```bash
swift run -c release media-generation-kit-cli auth login \
  --cloud-api-base-url "https://staging-api.drawthings.ai"
```

Auth state validation:

```bash
swift run -c release media-generation-kit-cli auth state \
  --api-key "API_KEY" \
  --cloud-api-base-url "https://api.drawthings.ai"
```

Auth token fetch:

```bash
swift run -c release media-generation-kit-cli auth token \
  --api-key "API_KEY"
```

Logout saved credentials:

```bash
swift run -c release media-generation-kit-cli auth logout
```

Notes:

- `auth state` and `auth token` may omit `--api-key` after `auth login`.
- If `auth login` used a custom `--cloud-api-base-url`, later saved-credential flows reuse that same API host unless explicitly overridden.

### Model Commands

Catalog list:

```bash
swift run -c release media-generation-kit-cli models list \
  --models-dir /tmp
```

Ensure model files exist locally:

```bash
swift run -c release media-generation-kit-cli models ensure \
  --models-dir /tmp \
  --model "flux_2_klein_4b_q8p.ckpt"
```

Inspect resolved model metadata:

```bash
swift run -c release media-generation-kit-cli models inspect \
  --models-dir /tmp \
  --model "hf://black-forest-labs/FLUX.2-klein-4B"
```

Catalog behavior:

- The CLI uses the async `MediaGenerationEnvironment` catalog APIs.
- `generate`, `models list`, and `models inspect` may populate metadata from bundled or remote catalog data when needed.

Known gap:

- `models list-remote` is still exposed, but it intentionally fails because the current public `MediaGenerationKit` API does not provide remote model listing yet.

### LoRA Commands

Convert LoRA:

```bash
swift run -c release media-generation-kit-cli lora convert \
  --input /path/to/my_lora.safetensors \
  --output /tmp/my_lora_lora_f16.ckpt
```

Convert LoRA with derived output name in a chosen directory:

```bash
swift run -c release media-generation-kit-cli lora convert \
  --input /path/to/my_lora.safetensors \
  --output-dir /tmp \
  --scale 0.8
```

Upload converted LoRA:

```bash
swift run -c release media-generation-kit-cli lora upload \
  --input /path/to/my_lora_lora_f16.ckpt \
  --api-key "API_KEY"
```

Saved-login upload variant:

```bash
swift run -c release media-generation-kit-cli auth login

swift run -c release media-generation-kit-cli lora upload \
  --input /path/to/my_lora_lora_f16.ckpt
```

### Storage Commands

Known gap:

- `storage info` is still exposed, but it intentionally fails because the current public `MediaGenerationKit` API does not provide storage inspection yet.

## Guardrails

- Prefer running the built binary instead of `swift run` when you need perfectly clean stdout.
- Do not reintroduce positional models-directory arguments.
- Do not add separate `generate remote` or `generate cloud-compute` subcommands.
- Do not document removed SDK types such as `GenerationPipeline`, `GenerationBackend`, `GenerationRequest`, `GenerationOptions`, or `CloudSession`.
- Do not reintroduce the removed legacy façade or the old internal runtime naming scheme into the public `MediaGenerationKit` API.

## License

`MediaGenerationKit` is distributed under LGPLv3.

For the public Swift package, the code and dependencies that cross over from
`draw-things-community` into the `media-generation-kit` distribution are relicensed under LGPLv3
as part of that package.

The intent of using LGPLv3 here is to encourage improvements to `MediaGenerationKit` and closely
related package code to be contributed back to `MediaGenerationKit`, not to force GPLv3-style
licensing onto downstream applications that use the package.
