# Cloud Compute

Use ``MediaGenerationPipeline/Backend/cloudCompute(apiKey:options:)`` to run against Draw Things cloud compute.

```swift
import Foundation
import UniformTypeIdentifiers
import MediaGenerationKit

@main
struct ExampleApp {
  static func main() async throws {
    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      "flux_2_klein_4b_q8p.ckpt",
      backend: .cloudCompute(apiKey: "dk_xxx")
    )

    pipeline.configuration.width = 1024
    pipeline.configuration.height = 1024
    pipeline.configuration.steps = 4

    let results = try await pipeline.generate(
      prompt: "a red cube on a table"
    )

    try results[0].write(
      to: URL(fileURLWithPath: "/tmp/cloud-output.png"),
      type: .png
    )
  }
}
```

## App Verification

``AppCheckConfiguration`` lets you attach additional verification tokens:

- `.none`
- `.firebase(token:)`
- `.supabase(token:)`

These options are configured through ``MediaGenerationPipeline/CloudComputeOptions``.
