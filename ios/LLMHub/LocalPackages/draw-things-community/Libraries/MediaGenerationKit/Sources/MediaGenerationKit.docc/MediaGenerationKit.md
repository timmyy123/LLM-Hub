# ``_MediaGenerationKit``

@Metadata {
  @DocumentationExtension(mergeBehavior: override)
}

Generate images from Swift using local models, a remote Draw Things server, or Draw Things cloud compute.

## Overview

`MediaGenerationKit` is centered around ``MediaGenerationPipeline``:

1. Create a pipeline with ``MediaGenerationPipeline/fromPretrained(_:backend:)``.
2. Adjust ``MediaGenerationPipeline/Configuration`` when you need to override recommended defaults.
3. Call ``MediaGenerationPipeline/generate(prompt:negativePrompt:inputs:stateHandler:)``.
4. Save each ``MediaGenerationPipeline/Result`` to disk.

When you pass a `stateHandler`, preview-bearing progress updates can also include
``MediaGenerationPipeline/Preview``.

The package also exposes:

- ``MediaGenerationEnvironment`` for model resolution, catalog inspection, and download/ensure workflows.
- ``LoRAImporter`` for local LoRA conversion.
- ``LoRAStore`` for cloud-side LoRA upload and deletion.

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:ModelResolutionAndDownloads>
- <doc:RemoteGeneration>
- <doc:CloudCompute>
- <doc:LoRAWorkflows>

### Main Symbols

- ``MediaGenerationPipeline``
- ``MediaGenerationPipeline/Preview``
- ``MediaGenerationEnvironment``
- ``MediaGenerationResolvedModel``
- ``LoRAImporter``
- ``LoRAStore``
- ``AppCheckConfiguration``
