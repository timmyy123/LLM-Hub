import DrawThingsCLILib
import Foundation

@main
struct DrawThingsCLIExecutable {
  static func main() {
    exit(DrawThingsCLI.run(arguments: Array(CommandLine.arguments.dropFirst())))
  }
}
