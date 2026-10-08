import ArgumentParser
import BinaryResources
import DataDogLog
import DataModels
import Dflat
import Diffusion
import Foundation
import GRPC
import GRPCControlPanelModels
import GRPCImageServiceModels
import GRPCServer
import ImageGenerator
import LocalImageGenerator
import Logging
import ModelZoo
import NIO
import NIOSSL
import NNC
import ProxyControlClient
import SQLiteDflat
import ServerLoRALoader
import Tokenizer
import Utils

private func createTemporaryDirectory() -> String {
  let fileManager = FileManager.default
  let tempDirURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
    UUID().uuidString)
  try! fileManager.createDirectory(
    at: tempDirURL, withIntermediateDirectories: true, attributes: nil)
  return tempDirURL.path
}

private func createLocalImageGenerator(queue: DispatchQueue) -> (String, LocalImageGenerator) {
  let tempDir = createTemporaryDirectory()
  let workspace = SQLiteWorkspace(
    filePath: "\(tempDir)/config.sqlite3", fileProtectionLevel: .noProtection)
  // Just have an empty query result for now
  let configurations = workspace.fetch(for: GenerationConfiguration.self).where(
    GenerationConfiguration.id == 0, limit: .limit(0))

  let tokenizerV1 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_json,
    merges: BinaryResources.merges_txt,
    textualInversions: [])
  let tokenizerV2 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_16e6_json,
    merges: BinaryResources.bpe_simple_vocab_16e6_txt,
    textualInversions: [])
  let tokenizerKandinsky = SentencePieceTokenizer(
    data: BinaryResources.xlmroberta_bpe_model, startToken: 0,
    endToken: 2, tokenShift: 1)

  let tokenizerXL = tokenizerV2

  let tokenizerT5 = SentencePieceTokenizer(
    data: BinaryResources.t5_spiece_model, startToken: nil,
    endToken: 1, tokenShift: 0)
  let tokenizerPileT5 = SentencePieceTokenizer(
    data: BinaryResources.pile_t5_spiece_model, startToken: nil,
    endToken: 2, tokenShift: 0)
  let tokenizerChatGLM3 = SentencePieceTokenizer(
    data: BinaryResources.chatglm3_spiece_model, startToken: nil,
    endToken: nil, tokenShift: 0)
  let tokenizerLlama3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_llama3_json, merges: BinaryResources.merges_llama3_txt,
    specialTokens: [
      "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
      "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
    ], unknownToken: "<|end_of_text|>", startToken: "<|begin_of_text|>",
    endToken: "<|end_of_text|>")
  let tokenizerUMT5 = SentencePieceTokenizer(
    data: BinaryResources.umt5_spiece_model, startToken: nil,
    endToken: 1, tokenShift: 0)
  let tokenizerQwen25 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen2_5_json, merges: BinaryResources.merges_qwen2_5_txt,
    specialTokens: [
      "</tool_call>": 151658, "<tool_call>": 151657, "<|box_end|>": 151649, "<|box_start|>": 151648,
      "<|endoftext|>": 151643, "<|file_sep|>": 151664, "<|fim_middle|>": 151660,
      "<|fim_pad|>": 151662, "<|fim_prefix|>": 151659, "<|fim_suffix|>": 151661,
      "<|im_end|>": 151645, "<|im_start|>": 151644, "<|image_pad|>": 151655,
      "<|object_ref_end|>": 151647, "<|object_ref_start|>": 151646, "<|quad_end|>": 151651,
      "<|quad_start|>": 151650, "<|repo_name|>": 151663, "<|video_pad|>": 151656,
      "<|vision_end|>": 151653, "<|vision_pad|>": 151654, "<|vision_start|>": 151652,
    ], unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|endoftext|>")
  let tokenizerQwen3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen3_json, merges: BinaryResources.merges_qwen3_txt,
    specialTokens: [
      "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
      "<|object_ref_start|>": 151646, "<|object_ref_end|>": 151647, "<|box_start|>": 151648,
      "<|box_end|>": 151649, "<|quad_start|>": 151650, "<|quad_end|>": 151651,
      "<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|vision_pad|>": 151654,
      "<|image_pad|>": 151655, "<|video_pad|>": 151656, "<tool_call>": 151657,
      "</tool_call>": 151658, "<|fim_prefix|>": 151659, "<|fim_middle|>": 151660,
      "<|fim_suffix|>": 151661, "<|fim_pad|>": 151662, "<|repo_name|>": 151663,
      "<|file_sep|>": 151664, "<tool_response>": 151665, "</tool_response>": 151666,
      "<think>": 151667, "</think>": 151668, "<|boi_token|>": 151669,
      "<|bor_token|>": 151670, "<|eor_token|>": 151671, "<|bot_token|>": 151672,
      "<|tms_token|>": 151673,
    ], unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|endoftext|>")
  let tokenizerQwen3MinimaxH3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen3_json, merges: BinaryResources.merges_qwen3_txt,
    specialTokens: [
      "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
      "<|object_ref_start|>": 151646, "<|object_ref_end|>": 151647, "<|box_start|>": 151648,
      "<|box_end|>": 151649, "<|quad_start|>": 151650, "<|quad_end|>": 151651,
      "<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|vision_pad|>": 151654,
      "<|image_pad|>": 151655, "<|video_pad|>": 151656, "<tool_call>": 151657,
      "</tool_call>": 151658, "<|fim_prefix|>": 151659, "<|fim_middle|>": 151660,
      "<|fim_suffix|>": 151661, "<|fim_pad|>": 151662, "<|repo_name|>": 151663,
      "<|file_sep|>": 151664, "<tool_response>": 151665, "</tool_response>": 151666,
      "<think>": 151667, "</think>": 151668,
      "<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671,
      "<|lyrics_start|>": 151672, "<|lyrics_end|>": 151673,
      "<|caption_start|>": 151674, "<|caption_end|>": 151675,
    ], unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|endoftext|>")
  let tokenizerMistral3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_mistral3_json, merges: BinaryResources.merges_mistral3_txt,
    specialTokens: [
      "<unk>": 0, "<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[AVAILABLE_TOOLS]": 5,
      "[/AVAILABLE_TOOLS]": 6, "[TOOL_RESULTS]": 7, "[/TOOL_RESULTS]": 8, "[TOOL_CALLS]": 9,
      "[IMG]": 10, "<pad>": 11, "[IMG_BREAK]": 12, "[IMG_END]": 13, "[PREFIX]": 14, "[MIDDLE]": 15,
      "[SUFFIX]": 16, "[SYSTEM_PROMPT]": 17, "[/SYSTEM_PROMPT]": 18, "[TOOL_CONTENT]": 19,
    ], unknownToken: "<unk>", startToken: "<s>", endToken: "</s>")
  let tokenizerGemma3 = SentencePieceTokenizer(
    data: BinaryResources.gemma3_spiece_model, startToken: 2, endToken: nil, tokenShift: 0)
  return (
    tempDir,
    LocalImageGenerator(
      queue: queue, configurations: configurations, workspace: workspace, tokenizerV1: tokenizerV1,
      tokenizerV2: tokenizerV2, tokenizerXL: tokenizerXL, tokenizerKandinsky: tokenizerKandinsky,
      tokenizerT5: tokenizerT5, tokenizerPileT5: tokenizerPileT5,
      tokenizerChatGLM3: tokenizerChatGLM3, tokenizerLlama3: tokenizerLlama3,
      tokenizerUMT5: tokenizerUMT5, tokenizerQwen25: tokenizerQwen25,
      tokenizerQwen3: tokenizerQwen3, tokenizerQwen3MinimaxH3: tokenizerQwen3MinimaxH3,
      tokenizerMistral3: tokenizerMistral3,
      tokenizerGemma3: tokenizerGemma3
    )
  )
}

#if os(Linux)
  private func localhost() -> String? {
    let maxLength = Int(HOST_NAME_MAX) + 1
    var buffer = [CChar](repeating: 0, count: maxLength)

    if gethostname(&buffer, maxLength) == 0 {
      return String(cString: buffer)
    } else {
      return nil
    }
  }

  private func checkWIFEXITED(_ status: Int32) -> Bool {
    return (status & 0x7f) == 0
  }

  private func getWEXITSTATUS(_ status: Int32) -> Int32 {
    return (status >> 8) & 0xff
  }

  private func checkWIFSIGNALED(_ status: Int32) -> Bool {
    return ((status & 0x7f) + 1) >> 1 > 0
  }

  private func getWTERMSIG(_ status: Int32) -> Int32 {
    return status & 0x7f
  }

  private func supervise(
    maxCrashesWithinTimeWindow: Int?, timeWindow: TimeInterval, childWork: () -> Void
  ) {
    var restartCount = 0
    var crashTimes: [Date] = []

    while true {
      let pid = fork()
      guard pid != 0 else {
        childWork()
        return
      }
      guard pid > 0 else {
        print("Fork failed")
        return
      }
      // Parent process
      var status: Int32 = 0
      waitpid(pid, &status, 0)
      let crashTime = Date()
      var shouldRestart = false

      if checkWIFEXITED(status) {
        let exitStatus = getWEXITSTATUS(status)
        if exitStatus == 0 {
          print("Child process exited normally")
          return
        }
        print("Child process exited with status: \(exitStatus)")
        shouldRestart = true
      } else if checkWIFSIGNALED(status) {
        print("Child process terminated by signal: \(getWTERMSIG(status))")
        shouldRestart = true
      }

      if shouldRestart {
        if let maxCrashesWithinTimeWindow = maxCrashesWithinTimeWindow {
          // Add current crash time to the list
          crashTimes.append(crashTime)

          // Remove crash times older than 1 minute
          crashTimes = crashTimes.filter { crashTime.timeIntervalSince($0) <= timeWindow }

          // Check if we've exceeded the crash limit
          if crashTimes.count >= maxCrashesWithinTimeWindow, let time = crashTimes.last {
            print(
              "ERROR: Too many crashes (\(crashTimes.count)) within \(timeWindow) seconds. Stopping auto-restart to prevent infinite crash loop."
            )
            print("Last \(crashTimes.count) crashes occurred at:")
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
            print("\(formatter.string(from: time))")
            print("ERROR: Too many crashes. Exiting with error code 2 for external supervisor.")
            exit(2)
          }
        }

        restartCount += 1
        print("Restarting... (Attempt \(restartCount), \(crashTimes.count) crashes in last minute)")
      }
    }
  }
#endif

enum CLIError: Error {
  case invalidModelPath
}

// Server info structure matching ServerArgu
struct ServerInfo: Codable {
  let address: String
  let port: Int
  let priority: Int
}

// Main configuration structure
struct AddServersConfiguration: Codable {
  var host: String
  var port: Int
  // Server management
  var servers: [ServerInfo]
}

extension AddServersConfiguration {
  static func parse(_ jsonString: String) throws -> AddServersConfiguration {
    guard let jsonData = jsonString.data(using: .utf8) else {
      throw ConfigurationError.invalidJSON
    }
    let decoder = JSONDecoder()
    return try decoder.decode(AddServersConfiguration.self, from: jsonData)
  }

}

enum ConfigurationError: Error {
  case invalidJSON
  case encodingError
}

@main
struct gRPCServerCLI: ParsableCommand {
  static let configuration: CommandConfiguration = CommandConfiguration(
    commandName: "gRPCServerCLI")
  @Argument(help: "The directory path to load models.")
  var modelsDirectory: String

  @Option(name: .shortAndLong, help: "The name you exposes in the local network.")
  var name: String = {
    #if os(Linux)
      if let localhost = localhost() {
        return localhost
      }
    #else
      if let handle = dlopen(
        "/System/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration", RTLD_LAZY)
      {
        defer {
          dlclose(handle)
        }
        if let sym = dlsym(handle, "SCDynamicStoreCopyComputerName") {
          let SCDynamicStoreCopyComputerName = unsafeBitCast(
            sym,
            to: (@convention(c) (OpaquePointer?, UnsafeMutablePointer<Unmanaged<CFString>?>?) ->
              Unmanaged<CFString>?).self)
          let name = SCDynamicStoreCopyComputerName(nil, nil)?.takeRetainedValue() as String? ?? ""
          return name
        }
      }
    #endif
    return "Unknown Computer"
  }()

  @Option(name: .shortAndLong, help: "The address in your local network that you want to expose.")
  var address: String = "0.0.0.0"

  @Option(name: .shortAndLong, help: "The port in your local network that you want to expose.")
  var port: Int = 7859

  @Option(name: .shortAndLong, help: "The GPU to use for image generation.")
  var gpu: Int = 0

  @Option(name: .shortAndLong, help: "Use Datadog as the logging backend.")
  var datadogAPIKey: String = ""

  @Option(name: .shortAndLong, help: "The shared secret that can help to secure the server.")
  var sharedSecret: String = ""

  @Flag(help: "Disable TLS for the connection.")
  var noTLS = false

  @Flag(help: "Disable response compression.")
  var noResponseCompression = false

  @Flag(help: "Enable model browsing.")
  var modelBrowser = false

  @Flag(help: "Disable FlashAttention.")
  var noFlashAttention = false

  @Option(name: .shortAndLong, help: "The weights cache size in GiB.")
  var weightsCache: Int = 0

  @Flag(help: "Debug flag for the verbose model inference logging.")
  var debug = false

  @Option(
    name: .long,
    help:
      "Path to the directory for secondary models, typically for ones downloaded from blob store.")
  var secondaryModelsDirectory: String?

  @Option(name: .long, help: "Blob store (typically AWS S3 or Cloudflare R2) API access key.")
  var blobStoreAccessKey: String?

  @Option(name: .long, help: "Blob store (typically AWS S3 or Cloudflare R2) API secret.")
  var blobStoreSecret: String?

  @Option(name: .long, help: "Endpoint URL for the blob store.")
  var blobStoreEndpoint: String?

  @Option(name: .long, help: "The bucket for the blob store.")
  var blobStoreBucket: String?

  @Flag(help: "Offload some weights to CPU during inference.")
  var cpuOffload = false

  @Flag(help: "Prefer to fread by overriding system preferences.")
  var freadPreferred = false

  @Option(
    name: .long,
    help: "The directory where some cached files can be stored."
  )
  var cacheUri: String?

  #if os(Linux)
    @Flag(help: "Supervise the server so it restarts upon a internal crash.")
    var supervised = false

    @Option(
      name: .long,
      help: "The maximum number of crashes within a given time window before we give up rebooting.")
    var maxCrashesWithinTimeWindow: Int?

    @Option(name: .long, help: "The time window for us to record number of crashes in seconds.")
    var crashTimeWindow: TimeInterval = 60.0

  #endif

  @Option(
    name: .long,
    help:
      "Inline JSON configuration string, example  --join '{\"host\":\"proxy ip\" , \"port\": proxy port, \"servers\": [{\"address\":\"gpu ip\", \"port\":gpu port, \"priority\":1 1 = high, 2 = low}]}' "
  )
  var join: String?

  @Option(
    name: .long,
    help:
      "When a request is done, generation should be returned timely. This warns us if it is not returned after specified seconds."
  )
  var cancellationWarningTimeout: TimeInterval? = nil

  @Option(
    name: .long,
    help:
      "When a request is done, generation should be returned timely. This crashes the process if it is not returned after specified seconds after the warning."
  )
  var cancellationCrashTimeout: TimeInterval? = nil
  @Flag(
    help:
      "Echo response will be done on the media generation queue. This helps to use echo call as a health check mechanism."
  )
  var echoOnQueue: Bool = false

  mutating func run() throws {
    #if os(Linux)
      if supervised {
        // Run in supervised mode
        supervise(
          maxCrashesWithinTimeWindow: maxCrashesWithinTimeWindow, timeWindow: crashTimeWindow
        ) {
          try? startGRPCServer()
        }
      } else {
        try startGRPCServer()
      }
    #else
      try startGRPCServer()
    #endif
  }

  func startGRPCServer() throws {
    if debug {
      DynamicGraph.logLevel = .verbose
    }

    let fileManager = FileManager.default
    var isDirectory: ObjCBool = false
    let exists = fileManager.fileExists(
      atPath: URL(fileURLWithPath: modelsDirectory).path, isDirectory: &isDirectory)

    if exists && isDirectory.boolValue {
    } else {
      throw CLIError.invalidModelPath
    }

    if !datadogAPIKey.isEmpty {
      let gpu = gpu
      let datadogAPIKey = datadogAPIKey
      let name = name

      LoggingSystem.bootstrap { label in
        // Create a multiplexing log handler that will send logs to multiple destinations
        var handlers: [LogHandler] = []

        // Always add the console log handler for local visibility
        handlers.append(StreamLogHandler.standardOutput(label: label))

        // If Datadog API key is provided, add the Datadog handler
        var datadogHandler = DataDogLogHandler(
          label: label,
          key: datadogAPIKey,
          hostname: "\(name) GPU \(gpu)",
          region: .US5
        )
        datadogHandler.metadata = ["gpu": "\(gpu)"]
        handlers.append(datadogHandler)

        // Return a multiplexing handler that writes to all configured handlers
        return MultiplexLogHandler(handlers)
      }
    }

    let serverLoRALoader: ServerLoRALoader?
    // Add the custom models directory if provided
    if let secondaryModelsDirectory = secondaryModelsDirectory, let accessKey = blobStoreAccessKey,
      let secret = blobStoreSecret, let blobStoreBucket = blobStoreBucket,
      let blobStoreEndpoint = blobStoreEndpoint
    {

      let r2Client = R2Client(
        accessKey: accessKey, secret: secret, endpoint: blobStoreEndpoint, bucket: blobStoreBucket)
      print("Create R2Client: \(r2Client)")

      let localLoRAManager = LocalLoRAManager(
        r2Client: r2Client, localDirectory: secondaryModelsDirectory)
      serverLoRALoader = ServerLoRALoader(localLoRAManager: localLoRAManager)
      // Check if directory exists (either it was there already or we created it)
      if fileManager.fileExists(atPath: secondaryModelsDirectory, isDirectory: &isDirectory),
        isDirectory.boolValue
      {
        print("Using another models directory: \(secondaryModelsDirectory)")
      } else {
        print(
          "Warning: Provided models directory path is not a valid directory: \(secondaryModelsDirectory)"
        )
        throw CLIError.invalidModelPath
      }
      ModelZoo.externalUrls = [
        URL(fileURLWithPath: modelsDirectory), URL(fileURLWithPath: secondaryModelsDirectory),
      ]
    } else {
      serverLoRALoader = nil
      ModelZoo.externalUrls = [URL(fileURLWithPath: modelsDirectory)]
    }

    if noFlashAttention {
      DeviceCapability.isMFAEnabled.store(0, ordering: .releasing)
    }
    if gpu > 0 && gpu < DeviceKind.GPUs.count {
      DeviceKind.GPUs.permute(gpu)
    }
    if cpuOffload {
      DeviceCapability.memoryCapacity = .medium  // This will trigger logic to offload some weights to CPU during inference.
    }
    if freadPreferred {
      DeviceCapability.isFreadPreferred = true  // This will not do mmap but use fread when needed.
    }
    DeviceCapability.maxTotalWeightsCacheSize = UInt64(weightsCache) * 1_024 * 1_024 * 1_024
    try self.runAndBlock(
      name: name, address: address, port: port, TLS: !noTLS,
      serverLoRALoader: serverLoRALoader)
  }

  func runAndBlock(
    name: String, address: String, port: Int, TLS: Bool,
    serverLoRALoader: ServerLoRALoader?
  ) throws {

    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    defer {
      try! group.syncShutdownGracefully()
    }
    let queue = DispatchQueue(label: "com.draw-things.edit", qos: .userInteractive)
    let (tempDir, localImageGenerator) = createLocalImageGenerator(queue: queue)
    if let cacheUri = cacheUri {
      DeviceCapability.cacheUri = URL(fileURLWithPath: cacheUri)
    } else {
      DeviceCapability.cacheUri = URL(fileURLWithPath: tempDir)
    }
    let cancellationMonitor: ImageGenerationServiceImpl.CancellationMonitor?
    if let cancellationWarningTimeout = cancellationWarningTimeout,
      let cancellationCrashTimeout = cancellationCrashTimeout
    {
      cancellationMonitor = ImageGenerationServiceImpl.CancellationMonitor(
        warningTimeout: cancellationWarningTimeout, crashTimeout: cancellationCrashTimeout)
    } else {
      cancellationMonitor = nil
    }
    let imageGenerationServiceImpl = ImageGenerationServiceImpl(
      imageGenerator: localImageGenerator, queue: queue, backupQueue: queue,
      serverConfigurationRewriter: serverLoRALoader,
      cancellationMonitor: cancellationMonitor, echoOnQueue: echoOnQueue)
    if noResponseCompression {
      imageGenerationServiceImpl.responseCompression.store(false, ordering: .releasing)
    } else {
      imageGenerationServiceImpl.responseCompression.store(true, ordering: .releasing)
    }
    if modelBrowser {
      imageGenerationServiceImpl.enableModelBrowsing.store(true, ordering: .releasing)
    } else {
      imageGenerationServiceImpl.enableModelBrowsing.store(false, ordering: .releasing)
    }
    imageGenerationServiceImpl.sharedSecret = sharedSecret.isEmpty ? nil : sharedSecret

    // Bind the server and get an `EventLoopFuture<Server>`
    let serverFuture: EventLoopFuture<GRPC.Server>
    let certificate = try? NIOSSLCertificate(
      bytes: [UInt8](BinaryResources.server_crt_crt), format: .pem)
    let privateKey = try? NIOSSLPrivateKey(
      bytes: [UInt8](BinaryResources.server_key_key), format: .pem)
    var TLS = TLS
    if TLS, let certificate = certificate, let privateKey = privateKey {
      serverFuture = GRPC.Server.usingTLS(
        with: GRPCTLSConfiguration.makeServerConfigurationBackedByNIOSSL(
          certificateChain: [.certificate(certificate)], privateKey: .privateKey(privateKey)),
        on: group
      )
      .withServiceProviders([imageGenerationServiceImpl])
      .withMaximumReceiveMessageLength(1024 * 1024 * 1024)
      .bind(host: address, port: port)
    } else {
      serverFuture = GRPC.Server.insecure(group: group)
        .withServiceProviders([imageGenerationServiceImpl])
        .withMaximumReceiveMessageLength(1024 * 1024 * 1024)
        .bind(host: address, port: port)
      TLS = false
    }

    let server: Server = try serverFuture.wait()

    if let localAddress = server.channel.localAddress, let port = localAddress.port {
      print("Server started on local address: \(localAddress), port \(port)")
    } else {
      print("Server started, but port is unknown")
    }
    let advertiser: GRPCServerAdvertiser = GRPCServerAdvertiser(name: name)
    advertiser.startAdvertising(port: Int32(port), TLS: TLS)

    if let json = join {
      try addServers(try AddServersConfiguration.parse(json))
    }

    // Block the current thread until the server closes
    try server.onClose.wait()
    advertiser.stopAdvertising()
  }

  func addServers(_ addServers: AddServersConfiguration) throws {
    guard !addServers.servers.isEmpty else {
      return
    }
    let client = ProxyControlClient()
    try client.connect(host: addServers.host, port: addServers.port)
    let group = DispatchGroup()  // TODO: I would prefer not use DispatchGroup here.
    for server in addServers.servers {
      group.enter()
      client.addGPUServer(
        address: server.address, port: server.port,
        isHighPriority: server.priority == 1
      ) {
        _ in
        group.leave()
      }
    }
    group.wait()
  }
}
