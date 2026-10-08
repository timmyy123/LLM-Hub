import Foundation

// Function to convert filename to valid C identifier
func sanitizeFilename(_ filename: String) -> String {
  return filename.replacingOccurrences(of: ".", with: "_")
    .replacingOccurrences(of: "-", with: "_")
    .replacingOccurrences(of: " ", with: "_")
}

// Function to read file and generate C array
func generateCArray(fromFile path: String) throws -> (String, String) {
  // Read file data
  let fileData = try Data(contentsOf: URL(fileURLWithPath: path))
  let filename = (path as NSString).lastPathComponent
  let sanitizedName = sanitizeFilename(filename)

  // Convert to bytes array
  let bytes = [UInt8](fileData)

  // Generate output with both array and size constant
  var output = "// Generated from: \(filename)\n\n#include <stddef.h>\n\n"

  // Add size constant first
  output += "const size_t \(sanitizedName)_size = \(bytes.count);\n\n"

  // Generate C array declaration
  output += "static const unsigned char \(sanitizedName)_data[] = {\n    "

  // Add bytes with formatting
  for (index, byte) in bytes.enumerated() {
    output += "\(Int(byte))"

    // Add separator unless it's the last byte
    if index < bytes.count - 1 {
      output += ", "
    }

    // Add newline every 12 bytes for readability
    if (index + 1) % 12 == 0 {
      output += "\n    "
    }
  }

  output += "\n};\n\n"
  output += "void *\(sanitizedName)() { return (void*)\(sanitizedName)_data; }\n"

  return (
    output,
    """
    #ifndef GENERATED_\(sanitizedName)_H
    #define GENERATED_\(sanitizedName)_H
    #include <stddef.h>

    #ifdef __cplusplus
    extern "C" {
    #endif
    extern const size_t \(sanitizedName)_size;
    extern void *\(sanitizedName)(void);
    #ifdef __cplusplus
    }
    #endif
    #endif
    """
  )
}

// Main execution
func main() {
  guard CommandLine.arguments.count > 1 else {
    print("Usage: \((CommandLine.arguments[0] as NSString).lastPathComponent) <input_file>")
    exit(1)
  }

  let inputFile = CommandLine.arguments[1]
  let outputFile = CommandLine.arguments[2]
  let outputHeader = CommandLine.arguments[3]

  do {
    let (cArray, header) = try generateCArray(fromFile: inputFile)

    // Write to output file
    try cArray.write(toFile: outputFile, atomically: true, encoding: .utf8)

    try header.write(toFile: outputHeader, atomically: true, encoding: .utf8)

    print("Successfully converted '\(inputFile)' to '\(outputFile)'")
    print(
      "Generated array size: \((try Data(contentsOf: URL(fileURLWithPath: inputFile))).count) bytes"
    )
  } catch {
    print("Error: \(error.localizedDescription)")
    exit(1)
  }
}

main()
