#!/bin/bash
# Generate DataModels Swift files from .fbs schemas using Bazel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

libraries=(DataModels History)
schema_targets=(
    //Libraries/DataModels:config_schema
    //Libraries/DataModels:estimation_schema
    //Libraries/DataModels:mixing_schema
    //Libraries/DataModels:lora_trainer_schema
    //Libraries/DataModels:dataset_schema
    //Libraries/DataModels:paint_color_schema
    //Libraries/DataModels:peer_connection_id_schema
    //Libraries/History:text_history_schema
)

# These app libraries are only present in the full repository.
if [[ -d "$PROJECT_ROOT/Libraries/ProjectHistoryManager" ]]; then
    libraries+=(ProjectHistoryManager)
    schema_targets+=(//Libraries/ProjectHistoryManager:project_history_schema)
fi
if [[ -d "$PROJECT_ROOT/Libraries/UserAccount" ]]; then
    libraries+=(UserAccount)
    schema_targets+=(
        //Libraries/UserAccount:account_schema
        //Libraries/UserAccount:privacy_pass_schema
    )
fi

echo "Building DataModels schemas with Bazel..."
bazel build "${schema_targets[@]}"

echo "Copying generated files to the SwiftPM source directories..."
output_dirs=()
for library in "${libraries[@]}"; do
    output_dir="$PROJECT_ROOT/Libraries/$library/PreGeneratedSPM"
    output_dirs+=("$output_dir")
    rm -rf "$output_dir"
    mkdir -p "$output_dir"
    # Copy all generated Swift files (skip JSON files).
    find "bazel-bin/Libraries/$library" -maxdepth 1 -name "*_generated.swift" \
        -exec cp {} "$output_dir/" \;
done

echo "Formatting generated Swift files with swift-format..."
swift_files=()
while IFS= read -r -d '' file; do
    swift_files+=("$file")
done < <(find "${output_dirs[@]}" -name "*.swift" -print0)

if [[ ${#swift_files[@]} -gt 0 ]]; then
    bazel_swift_format_args=(--compilation_mode=opt)
    if [[ "$(uname)" == "Darwin" ]]; then
        export DYLD_FALLBACK_LIBRARY_PATH="$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx"
        # Keep the formatter binary runnable on CI hosts older than the current Xcode SDK.
        bazel_swift_format_args+=(--macos_minimum_os=15.0)
    fi
    bazel run "${bazel_swift_format_args[@]}" @SwiftFormat//:swift-format -- \
        format --configuration "$PROJECT_ROOT/.swift-format.json" -i "${swift_files[@]}"
fi

echo "Generated files:"
ls -la "${output_dirs[@]}"

echo "Done!"
