# Getting Started

The smallest workflow to run generation locally.

```swift
import Foundation
import UniformTypeIdentifiers
import MediaGenerationKit

@main
struct ExampleApp {
  static func main() async throws {
    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      "hf://black-forest-labs/FLUX.2-klein-4B",
      backend: .local
    )

    pipeline.configuration.width = 1024
    pipeline.configuration.height = 1024
    pipeline.configuration.steps = 4

    let results = try await pipeline.generate(
      prompt: "a red cube on a table"
    )

    try results[0].write(
      to: URL(fileURLWithPath: "/tmp/output.png"),
      type: .png
    )
  }
}
```

## Recommended Defaults

`MediaGenerationKit` starts each pipeline from the model's recommended template. For many models, you only need to override a few fields:

- `width`
- `height`
- `steps`
- `strength` for image-to-image
- `loras`
- `controls`

For `FLUX.2 [klein] 4B`, the normal example settings are:

- `width = 1024`
- `height = 1024`
- `steps = 4`

## Image Inputs

Use ``MediaGenerationPipeline/file(_:)`` or ``MediaGenerationPipeline/data(_:)`` for encoded images:

```swift
pipeline.configuration.strength = 0.35

let results = try await pipeline.generate(
  prompt: "studio portrait",
  inputs: [
    MediaGenerationPipeline.file("/tmp/input.png")
  ]
)
```

Role-specific wrappers let you provide masks and hints:

- ``MediaGenerationPipeline/MaskInput``
- ``MediaGenerationPipeline/MoodboardInput``
- ``MediaGenerationPipeline/DepthInput``

## Progress And Previews

`generate(...)` can report both progress state and an optional live preview:

```swift
let results = try await pipeline.generate(
  prompt: "a red cube on a table"
) { state, preview in
  if case .generating(let step, let totalSteps) = state {
    print("sampling \\(step)/\\(totalSteps)")
  }
  if let image = preview?[0] {
    print("received preview image: \\(image.width)x\\(image.height)")
  }
}
```
