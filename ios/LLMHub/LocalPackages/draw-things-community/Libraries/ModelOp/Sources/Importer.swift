import Dispatch

public enum Importer {
  public static let queue = DispatchQueue(
    label: "com.draw-things.model-import", qos: .userInitiated)
}

extension Importer {
  public static func cleanup(filename: String) -> String {
    // Convert the model name into file name.
    var finalName = ""
    for character in filename {
      if character.isASCII && (character.isLetter || character.isNumber || character == ".") {
        finalName += String(character)
      } else {
        finalName += "_"
      }
    }
    return finalName.lowercased()
  }
}
