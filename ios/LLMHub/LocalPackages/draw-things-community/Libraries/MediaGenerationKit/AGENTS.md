# AGENTS.md

## Purpose

This guide is for agents working on the current `MediaGenerationKit` surface.

The package is meant to be sufficient to build a Swift app with:

- `MediaGenerationPipeline`
- `MediaGenerationEnvironment`
- `LoRAImporter`
- `LoRAStore`
- `MediaGenerationKitError`
- `AppCheckConfiguration`

Do not reintroduce the removed legacy façade or the old internal runtime naming scheme.

## Package Shape

- `MediaGenerationPipeline` is the main generation entry point.
- `MediaGenerationPipeline.fromPretrained(...)` is async.
- `pipeline.configuration` is the mutable generation configuration.
- `MediaGenerationEnvironment.default` owns process-wide defaults.
- `LoRAImporter` is for local LoRA conversion.
- `LoRAStore` is for Draw Things cloud LoRA storage.

## Canonical App Flow

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
      negativePrompt: "",
      inputs: []
    )

    try results[0].write(
      to: URL(fileURLWithPath: "/tmp/cat.png"),
      type: .png
    )
  }
}
```

## Backend Shape

- Local:
  - `backend: .local`
  - `backend: .local(directory: "/path/to/Models")`
- Remote:
  - `backend: .remote(.init(host: "127.0.0.1", port: 7859))`
- Cloud compute:
  - `backend: .cloudCompute(apiKey: "YOUR_API_KEY")`

Remote defaults should assume TLS unless explicitly disabled in remote options.

## Inputs And Results

Use `MediaGenerationPipeline.generate(prompt:negativePrompt:inputs:stateHandler:)`.

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

## Configuration Model

- Configuration lives on `pipeline.configuration`.
- Model identity is bound by `fromPretrained(...)`.
- Copied pipelines should not share mutable configuration.
- Async cancellation is `Swift.Task` cancellation.
- Public config is intentionally bridged through `JSGenerationConfiguration` internally to avoid drift.

## Environment Helpers

`MediaGenerationEnvironment.default` owns process-wide settings and model-management helpers.

Important members:

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

## LoRA Flows

Local conversion:

- `LoRAImporter(file:version:)`
- optional `inspect()`
- `import(to:scaleFactor:progressHandler:)`

Cloud storage:

- `LoRAStore(backend:)`
- `upload(_:file:)`
- `delete(_:)`
- `delete(keys:)`

Do not put LoRA upload/delete methods directly on `MediaGenerationPipeline.Backend`.

## Guardrails

- Keep the public surface value-oriented and small.
- Prefer async public APIs for any operation that may block on network or long-running work.
- Do not reintroduce `GenerationPipeline`, `GenerationBackend`, `GenerationTypes`, `CloudSession`, or `SDKRuntime*` names into the public API.
- Do not make remote/cloud behavior depend on `echo` shortcuts beyond the current handshake and cached cloud model list behavior.

## Verification

Minimum checks after API or internal runtime changes:

1. `bazel build //Libraries/MediaGenerationKit:MediaGenerationKit`
2. `bazel test //Libraries/MediaGenerationKit:MediaGenerationKitTests`
3. `bazel build //Apps:MediaGenerationKitCLI`
