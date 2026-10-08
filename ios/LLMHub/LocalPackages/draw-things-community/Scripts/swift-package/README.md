# Building with Swift Package Manager

This directory contains scripts for building DrawThings components using Swift Package Manager (SPM).

## Quick Start

```bash
# From project root:

# 1. Generate required source files (requires Bazel)
./Scripts/swift-package/generate_binary_resources.sh
./Scripts/swift-package/generate_datamodels.sh

# 2. Build
swift build --target gRPCServerCLI

# 3. Run
.build/debug/gRPCServerCLI /path/to/models
```

## Scripts

### generate_binary_resources.sh

Generates C source files from binary resources (vocab files, certificates, etc.) using Bazel's resource packager.

**Output:** `Libraries/BinaryResources/GeneratedC/`

**When to run:** After modifying files in `Libraries/BinaryResources/Resources/`

### generate_datamodels.sh

Generates Swift source files from FlatBuffer schemas using Bazel's dflatc tool.

**Output:** `Libraries/DataModels/PreGeneratedSPM/`

**When to run:** After modifying `.fbs` files in `Libraries/DataModels/Sources/`

## Build Commands

```bash
# Debug build
swift build --target gRPCServerCLI

# Release build
swift build -c release --target gRPCServerCLI

# Run directly
swift run gRPCServerCLI /path/to/models

# Show binary location
swift build --target gRPCServerCLI --show-bin-path
```

## Targets

| Target | Description |
|--------|-------------|
| `gRPCServerCLI` | gRPC server for image generation |

## Troubleshooting

### Missing generated files

If you see errors about missing files in `GeneratedC/` or `PreGeneratedSPM/`, run the generation scripts:

```bash
./Scripts/swift-package/generate_binary_resources.sh
./Scripts/swift-package/generate_datamodels.sh
```

## Notes

- The SPM build uses the same source files as Bazel where possible
- Generated files are committed to the repository for convenience
- Re-run generation scripts only when source schemas change
