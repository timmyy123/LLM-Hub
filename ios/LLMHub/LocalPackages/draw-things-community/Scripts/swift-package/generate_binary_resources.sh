#!/bin/bash
# Generate C resource files from binary resources using Bazel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUTPUT_DIR="$PROJECT_ROOT/Libraries/BinaryResources/GeneratedC"

cd "$PROJECT_ROOT"

# Build the Resources target
echo "Building BinaryResources with Bazel..."
bazel build //Libraries/BinaryResources:Resources

# Clear and copy generated files
echo "Copying generated files to $OUTPUT_DIR..."
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy only .c and .h files
cp bazel-bin/Libraries/BinaryResources/*_generated.c "$OUTPUT_DIR/"
cp bazel-bin/Libraries/BinaryResources/*_generated.h "$OUTPUT_DIR/"

# Create umbrella header
echo "Creating umbrella header..."
cat > "$OUTPUT_DIR/C_Resources.h" << 'EOF'
// Umbrella header for C_Resources module (SPM)
#ifndef C_RESOURCES_H
#define C_RESOURCES_H

EOF

for h in "$OUTPUT_DIR"/*_generated.h; do
    basename=$(basename "$h")
    echo "#include \"$basename\"" >> "$OUTPUT_DIR/C_Resources.h"
done

echo "" >> "$OUTPUT_DIR/C_Resources.h"
echo "#endif // C_RESOURCES_H" >> "$OUTPUT_DIR/C_Resources.h"

echo "Generated files:"
ls -la "$OUTPUT_DIR/"

echo "Done!"
