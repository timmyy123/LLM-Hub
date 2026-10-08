# Model Resolution and Downloads

``MediaGenerationEnvironment`` owns process-scoped helpers for model catalogs and download workflows.

## Resolve a Model

Async resolution is the network-capable path:

```swift
let model = await MediaGenerationEnvironment.default.resolveModel(
  "hf://black-forest-labs/FLUX.2-klein-4B"
)
```

Sync resolution is offline-only or cache-only:

```swift
let model = try MediaGenerationEnvironment.default.resolveModel(
  "flux_2_klein_4b_q8p.ckpt"
)
```

If a synchronous call would need to fetch remote catalog data, it throws `MediaGenerationKitError.asyncOperationRequired(...)`.

## Ensure a Model is Ready

Use ``MediaGenerationEnvironment/ensure(_:offline:stateHandler:)`` to resolve a model and make sure required local files are present:

```swift
let resolved = try await MediaGenerationEnvironment.default.ensure(
  "hf://black-forest-labs/FLUX.2-klein-4B"
) { state in
  print(state)
}
```

`ensure` includes dependencies, not just the primary checkpoint file.

## Catalog Inspection

The environment also exposes:

- `inspectModel`
- `downloadableModels`
- `suggestedModels`
