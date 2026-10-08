# LoRA Workflows

`MediaGenerationKit` separates local LoRA conversion from cloud-side LoRA storage.

## Convert a LoRA File

Use ``LoRAImporter`` to inspect and convert LoRA files into Draw Things format:

```swift
import Foundation
import MediaGenerationKit

var importer = LoRAImporter(
  file: URL(fileURLWithPath: "/tmp/source.safetensors")
)

try importer.inspect()

try importer.import(
  to: URL(fileURLWithPath: "/tmp/output_lora_f16.ckpt")
)
```

## Upload to Cloud

Use ``LoRAStore`` with a cloud compute backend:

```swift
import Foundation
import MediaGenerationKit

let store = try LoRAStore(
  backend: .cloudCompute(apiKey: "dk_xxx")
)

let data = try Data(contentsOf: URL(fileURLWithPath: "/tmp/output_lora_f16.ckpt"))
let uploaded = try await store.upload(data, file: "output_lora_f16.ckpt")
print(uploaded.file)
```

`LoRAStore.list()` is currently unavailable because the public cloud API does not expose a list endpoint.
