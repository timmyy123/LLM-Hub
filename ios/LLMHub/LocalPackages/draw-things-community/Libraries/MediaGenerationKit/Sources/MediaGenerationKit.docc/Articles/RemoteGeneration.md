# Remote Generation

Use ``MediaGenerationPipeline/Backend/remote(_:options:)`` when you want to run inference against your own Draw Things server.

```swift
import Foundation
import UniformTypeIdentifiers
import MediaGenerationKit

@main
struct ExampleApp {
  static func main() async throws {
    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      "flux_2_klein_4b_i8x.ckpt",
      backend: .remote(
        .init(host: "127.0.0.1", port: 7859)
      )
    )

    pipeline.configuration.width = 1024
    pipeline.configuration.height = 1024
    pipeline.configuration.steps = 4

    let results = try await pipeline.generate(
      prompt: "a red cube on a table"
    )

    try results[0].write(
      to: URL(fileURLWithPath: "/tmp/remote-output.png"),
      type: .png
    )
  }
}
```

## Notes

- TLS is enabled by default in ``MediaGenerationPipeline/RemoteOptions``.
- Use `sharedSecret` only when your remote server requires it.
- Generic remote execution does not depend on the remote `echo` model list for model existence checks.
