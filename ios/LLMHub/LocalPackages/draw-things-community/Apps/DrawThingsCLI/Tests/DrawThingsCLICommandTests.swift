import BashToolContext
import CLICloudAuth
import Downloader
import DrawThingsCLICommand
import DrawThingsCLILib
import Foundation
import ModelZoo
import ios_system

private enum TestFailure: Error {
  case assertion(String)
}

private func expect(_ value: Bool, line: Int = #line) throws {
  guard value else { throw TestFailure.assertion("Failed at line \(line)") }
}

private func expectEqual<T: Equatable>(_ actual: T, _ expected: T, line: Int = #line) throws {
  guard actual == expected else {
    throw TestFailure.assertion("Line \(line): expected \(expected), got \(actual)")
  }
}

private func unwrap<T>(_ value: T?) throws -> T {
  guard let value else { throw TestFailure.assertion("Unexpected nil") }
  return value
}

private final class TestToolAccountProvider: ToolAccountProvider {
  var cloudKey: String? = "dk_embedded_test"
  var resolutionCount = 0

  func resolveDrawThingsCredential(
    prepareIfNeeded: Bool, completion: @escaping (Result<String?, Error>) -> Void
  ) {
    try! expect(!prepareIfNeeded)
    resolutionCount += 1
    completion(.success(cloudKey))
  }

  func drawThingsInsufficientFunds(apiKey: String) {
    preconditionFailure("Auth status must not open a purchase flow")
  }
}

@main
struct DrawThingsCLICommandTests {
  static func main() throws {
    let session = strdup("draw-things-cli-tests")!
    ios_switchSession(session)
    initializeEnvironment()
    defer {
      ios_setStreams(stdin, stdout, stderr)
      ios_closeSession(session)
      free(session)
    }
    try expectEqual(ios_registerCommand("draw-things-cli", draw_things_main), 1)
    for command in ["draw-things-cli", "/usr/bin/draw-things-cli"] {
      try expectEqual(
        command.withCString { ios_commandIsExclusiveOrThreadSafe($0) },
        IOS_COMMAND_ISOLATION_EXCLUSIVE_DRAW_THINGS)
    }
    var resolutionCount = 0
    var unloadCount = 0
    var downloadStarts = 0
    var downloadFinishes = 0
    var generationStarts = 0
    var generationFinishes = 0
    var time: TimeInterval = 0
    var expectedUserTime: TimeInterval = 0
    var expectedSystemTime: TimeInterval = 0
    let interaction = BashUserInteraction(time: { time })
    let accountProvider = TestToolAccountProvider()
    let callerThread = Thread.current
    let previousContext = BashToolContext.current
    Thread.current.threadDictionary["draw-things-tests.unrelated"] = "do not inherit"
    BashToolContext.current = BashToolContext(
      accountProvider: accountProvider,
      resloveDrawThingsModelsDirectory: { requested, interaction, completion in
        try! expect(Thread.current !== callerThread)
        try! expect(Thread.current.threadDictionary["draw-things-tests.unrelated"] == nil)
        resolutionCount += 1
        let request = interaction.begin(isUserInteraction: true)
        time += 2
        expectedUserTime += 2
        expectedSystemTime += 2
        interaction.cancel(request)
        completion(.success(requested!))
      }, bashUserInteraction: interaction, unloadTextGenerator: { unloadCount += 1 },
      modelDownloadEvent: { event in
        switch event {
        case .started(_, _, _, _, let cancel):
          downloadStarts += 1
          time += 10
          expectedSystemTime += 10
          try! expectEqual(interaction.userInteractionInterval(since: 0), expectedUserTime)
          try! expectEqual(interaction.systemInteractionInterval(since: 0), expectedSystemTime)
          cancel()
        case .progress: break
        case .finished:
          downloadFinishes += 1
          time += 1
          try! expectEqual(interaction.userInteractionInterval(since: 0), expectedUserTime)
          try! expectEqual(interaction.systemInteractionInterval(since: 0), expectedSystemTime)
        }
      },
      imageGenerationEvent: { event in
        switch event {
        case .started(_, _, _, let prompt, _, let cancel):
          generationStarts += 1
          try! expectEqual(prompt, "A mountain lake")
          time += 20
          expectedSystemTime += 20
          try! expectEqual(interaction.userInteractionInterval(since: 0), expectedUserTime)
          try! expectEqual(interaction.systemInteractionInterval(since: 0), expectedSystemTime)
          cancel()
        case .finished:
          generationFinishes += 1
          time += 1
          try! expectEqual(interaction.userInteractionInterval(since: 0), expectedUserTime)
          try! expectEqual(interaction.systemInteractionInterval(since: 0), expectedSystemTime)
        case .progress, .segmentStarted:
          try! expect(false)
        }
      })
    // Per-agent and per-turn context copies must retain the same account provider.
    BashToolContext.current = BashToolContext.current?.with(unloadTextGenerator: {
      unloadCount += 1
    })
    weak var inheritedContext = BashToolContext.current
    defer {
      BashToolContext.current = previousContext
      Thread.current.threadDictionary.removeObject(forKey: "draw-things-tests.unrelated")
    }
    try withContext { context, directory, contents in
      let project = directory.appendingPathComponent("Project", isDirectory: true)
      try FileManager.default.createDirectory(at: project, withIntermediateDirectories: true)
      try expectEqual(
        ios_setSystemPaths(directory, directory, directory, directory, project, directory), 1)
      ios_setDirectoryURL(project)
      ios_setStreams(context.input, context.output, context.error)
      defer { ios_setStreams(stdin, stdout, stderr) }
      try expectEqual(ios_system_osh("draw-things-cli generate --help > help.txt; cat help.txt"), 0)
      try expect(contents().0.contains("--model"))
      try expectEqual(resolutionCount, 0)
      try expectEqual(unloadCount, 0)
      try expectEqual(ios_system_osh("draw-things-cli auth status"), 0)
      try expect(contents().0.contains("\"defaultBackend\":\"cloud\""))
      try expect(!contents().0.contains("dk_embedded_test"))
      accountProvider.cloudKey = nil
      try expectEqual(ios_system_osh("draw-things-cli auth status"), 0)
      try expect(contents().0.contains("\"defaultBackend\":\"local\""))
      try expectEqual(accountProvider.resolutionCount, 2)
      try expectEqual(resolutionCount, 0)
      try expectEqual(unloadCount, 0)
      try expectEqual(ios_system_osh("draw-things-cli --not-a-flag"), 64)
      try expect(contents().1.contains("--not-a-flag"))
      try expectEqual(
        ios_system_osh("draw-things-cli completion bash --output /home/Project/completion.sh"), 0)
      try expect(
        FileManager.default.fileExists(atPath: project.appendingPathComponent("completion.sh").path)
      )
      try expectEqual(
        ios_system_osh("DRAWTHINGS_MODELS_DIR=Models draw-things-cli models list --offline"), 0)
      try expect(
        FileManager.default.fileExists(atPath: project.appendingPathComponent("Models").path))
      try expectEqual(resolutionCount, 1)
      let script = project.appendingPathComponent("models.sh")
      try "draw-things-cli models list --offline --models-dir Models | cat\n".write(
        to: script, atomically: true, encoding: .utf8)
      // Launch sh directly: nesting a new OSH invocation inside an active OSH
      // session is intentionally rejected by ios_system. This still crosses
      // both the sh command thread and the script's pipeline command threads.
      let shellArguments = ["sh", "models.sh"].map { (value: String) in strdup(value) }
      defer { shellArguments.forEach { free($0) } }
      var shellPointers = shellArguments.map { $0.map { UnsafePointer($0) } }
      // sh owns its redirected streams and closes them on exit.
      let shellInput = try unwrap(fdopen(dup(fileno(context.input)), "r"))
      let shellOutput = try unwrap(fdopen(dup(fileno(context.output)), "a"))
      let shellError = try unwrap(fdopen(dup(fileno(context.error)), "a"))
      ios_setStreams(shellInput, shellOutput, shellError)
      let shellStatus = ios_system(Int32(shellPointers.count), &shellPointers)
      ios_setStreams(context.input, context.output, context.error)
      try expectEqual(shellStatus, 0)
      try expectEqual(resolutionCount, 2)
      try expectEqual(unloadCount, 0)
      try expectEqual(
        ios_system_osh(
          "printf 'test prompt' | draw-things-cli generate --offline --no-download-missing --models-dir Models --model flux_2_klein_4b_q6p.ckpt --prompt-file - --output result.png"
        ), 1)
      try expect(contents().1.contains("Missing model files:"))

      let downloadFile = "test-command-download-\(UUID().uuidString).ckpt"
      let originalOverrides = ModelZoo.overrideMapping
      ModelZoo.overrideMapping[downloadFile] = ModelZoo.Specification(
        name: "Test command download", file: downloadFile, prefix: "", version: .v1)
      let downloadStatus = ios_system_osh(
        "draw-things-cli models ensure --models-dir Models --model \(downloadFile) --no-include-dependencies"
      )
      ModelZoo.overrideMapping = originalOverrides
      try expectEqual(downloadStatus, 130)
      try expectEqual(downloadStarts, 1)
      try expectEqual(downloadFinishes, 1)
      try expect(contents().1.contains("Download model cancelled by user\n"))

      try expectEqual(
        ios_system_osh(
          "draw-things-cli generate --models-dir Models --remote --remote-url 127.0.0.1 --remote-port 1 --model flux_2_klein_4b_q6p.ckpt --prompt 'A mountain lake' --output result.png"
        ), 130)
      try expectEqual(generationStarts, 1)
      try expectEqual(generationFinishes, 1)
      try expect(contents().1.contains("Generation aborted by user.\n"))
      try expect(
        !FileManager.default.fileExists(atPath: project.appendingPathComponent("result.png").path))

      var descriptors: [Int32] = [0, 0]
      try expectEqual(pipe(&descriptors), 0)
      let input = try unwrap(fdopen(descriptors[0], "r"))
      defer {
        fclose(input)
        close(descriptors[1])
      }
      let finished = DispatchGroup()
      finished.enter()
      var cancelledStatus: Int32 = 0
      let commandContext = BashToolContext.current
      DispatchQueue.global().async {
        let previousContext = BashToolContext.current
        BashToolContext.current = commandContext
        defer { BashToolContext.current = previousContext }
        ios_switchSession(session)
        ios_setStreams(input, context.output, context.error)
        let strings: [String] = [
          "draw-things-cli", "generate", "--offline", "--models-dir", "Models",
          "--model", "flux_2_klein_4b_q6p.ckpt", "--prompt-file", "-", "--output", "result.png",
        ]
        let arguments = strings.map { strdup($0) }
        defer { arguments.forEach { free($0) } }
        var pointers = arguments.map { $0.map { UnsafePointer($0) } }
        cancelledStatus = ios_system(Int32(pointers.count), &pointers)
        finished.leave()
      }
      // Give the command time to enter its blocking stdin read, as in the
      // Z3 command test. No generation or network access is involved.
      usleep(200_000)
      try expectEqual(ios_kill(), 0)
      try expectEqual(finished.wait(timeout: .now() + 10), .success)
      try expectEqual(cancelledStatus, 130)
      ios_setStreams(context.input, context.output, context.error)
      try expectEqual(ios_system_osh("draw-things-cli --help"), 0)
      if let models = ProcessInfo.processInfo.environment["DRAWTHINGS_CLI_TEST_MODELS_DIR"] {
        ios_setenv("DRAWTHINGS_MODELS_DIR", models, 1)
        let model =
          ProcessInfo.processInfo.environment["DRAWTHINGS_CLI_TEST_MODEL"]
          ?? "sd_v1.5_f16.ckpt"
        ios_setenv("DRAWTHINGS_CLI_TEST_MODEL", model, 1)
        try expectEqual(
          ios_system_osh(
            "draw-things-cli generate --offline --model \"$DRAWTHINGS_CLI_TEST_MODEL\" --prompt 'a red cube' --width 64 --height 64 --steps 1 --cfg 1 --seed 42 --output generated.png"
          ), 0)
        let png = try Data(contentsOf: project.appendingPathComponent("generated.png"))
        try expectEqual(Array(png.prefix(8)), [137, 80, 78, 71, 13, 10, 26, 10])
        try expectEqual(unloadCount, 1)
      }
    }
    // Finished commands must not retain the context after its host releases it.
    BashToolContext.current = nil
    try expect(inheritedContext == nil)
  }

  private static func withContext(
    _ body: (DrawThingsCLIContext, URL, () -> (String, String)) throws -> Void
  ) throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let input = try unwrap(tmpfile())
    let output = try unwrap(tmpfile())
    let error = try unwrap(tmpfile())
    defer {
      fclose(input)
      fclose(output)
      fclose(error)
      try? FileManager.default.removeItem(at: directory)
    }
    let context = DrawThingsCLIContext(
      input: input, output: output, error: error,
      environment: ["DRAWTHINGS_MODELS_DIR": "Models"], isStandardOutputTTY: false,
      resolvePath: { URL(fileURLWithPath: $0, relativeTo: directory).standardizedFileURL.path })
    func contents(_ stream: UnsafeMutablePointer<FILE>) -> String {
      fflush(stream)
      rewind(stream)
      var bytes = [UInt8](repeating: 0, count: 4096)
      var data = Data()
      while true {
        let count = fread(&bytes, 1, bytes.count, stream)
        if count == 0 { break }
        data.append(contentsOf: bytes.prefix(count))
      }
      return String(decoding: data, as: UTF8.self)
    }
    do {
      try body(context, directory, { (contents(output), contents(error)) })
    } catch let failure {
      fputs(contents(error), stderr)
      throw failure
    }
  }

}
