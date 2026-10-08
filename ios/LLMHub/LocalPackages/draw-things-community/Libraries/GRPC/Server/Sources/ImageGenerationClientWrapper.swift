import Atomics
import BinaryResources
import Foundation
import GRPC
import GRPCImageServiceModels
import ModelZoo
import NIO
import NIOHTTP2
import NIOSSL

public final class ImageGenerationClientWrapper {
  public final class MonitoringHandler: ChannelDuplexHandler {
    public struct Statistics {
      public var bytesSent: Int
      public var bytesReceived: Int
      public init(bytesSent: Int, bytesReceived: Int) {
        self.bytesSent = bytesSent
        self.bytesReceived = bytesReceived
      }
    }
    public static var statistics: Statistics {
      return Statistics(
        bytesSent: bytesSent.load(ordering: .acquiring),
        bytesReceived: bytesReceived.load(ordering: .acquiring))
    }
    private static let bytesSent = ManagedAtomic<Int>(0)
    private static let bytesReceived = ManagedAtomic<Int>(0)

    public typealias InboundIn = ByteBuffer
    public typealias InboundOut = ByteBuffer
    public typealias OutboundIn = ByteBuffer
    public typealias OutboundOut = ByteBuffer

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
      // Unwrap the NIOAny to get a ByteBuffer
      let buffer = self.unwrapInboundIn(data)
      let byteCount = buffer.readableBytes
      Self.bytesReceived.wrappingIncrement(by: byteCount, ordering: .acquiringAndReleasing)
      // Forward the bytes we read
      context.fireChannelRead(data)
    }

    public func write(
      context: ChannelHandlerContext, data: NIOAny, promise: EventLoopPromise<Void>?
    ) {
      // Unwrap the NIOAny to get a ByteBuffer
      let buffer = self.unwrapOutboundIn(data)
      let byteCount = buffer.readableBytes
      Self.bytesSent.wrappingIncrement(by: byteCount, ordering: .acquiringAndReleasing)
      // Forward the bytes
      context.write(data, promise: promise)
    }
  }
  public enum Error: Swift.Error {
    case invalidRootCA
  }
  private var deviceName: String? = nil
  private var eventLoopGroup: EventLoopGroup? = nil
  private var channel: GRPCChannel? = nil
  public private(set) var sharedSecret: String? = nil
  public private(set) var client: ImageGenerationServiceNIOClient? = nil

  public init(deviceName: String? = nil) {
    self.deviceName = deviceName
  }

  public func connect(
    host: String, port: Int, TLS: Bool, hostnameVerification: Bool, sharedSecret: String?
  ) throws {
    try? eventLoopGroup?.syncShutdownGracefully()
    let eventLoopGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    let transportSecurity: GRPCChannelPool.Configuration.TransportSecurity
    if TLS {
      let bytes = [UInt8](BinaryResources.root_ca_crt)
      guard bytes.count > 0 else {
        throw ImageGenerationClientWrapper.Error.invalidRootCA
      }
      let certificate = try NIOSSLCertificate(bytes: bytes, format: .pem)

      let isrgrootx1 = {
        let bytes = [UInt8](BinaryResources.isrgrootx1_pem)
        return try? NIOSSLCertificate(bytes: bytes, format: .pem)
      }()

      transportSecurity = .tls(
        .makeClientConfigurationBackedByNIOSSL(
          trustRoots: .certificates([certificate] + (isrgrootx1.map { [$0] } ?? [])),
          certificateVerification: hostnameVerification
            ? .fullVerification : .noHostnameVerification
        ))
    } else {
      transportSecurity = .plaintext
    }
    var configuration = GRPCChannelPool.Configuration.with(
      target: .host(host, port: port),
      transportSecurity: transportSecurity,
      eventLoopGroup: eventLoopGroup)
    configuration.debugChannelInitializer = { channel in
      channel.eventLoop.makeCompletedFuture {
        let sync = channel.pipeline.syncOperations
        let http2Handler = try sync.handler(type: NIOHTTP2Handler.self)
        // Note: this closure is called for every new connection, so you should
        // emit any events to a shared `Sendable` object held by the `ByteRecordingHandler`.
        // That object will be the bridge between the connection and your application.
        let monitoringHandler = MonitoringHandler()
        try sync.addHandler(monitoringHandler, position: .before(http2Handler))
      }
    }
    configuration.maximumReceiveMessageLength = 1024 * 1024 * 1024
    let channel = try GRPCChannelPool.with(configuration: configuration)
    let client = ImageGenerationServiceNIOClient(channel: channel)
    self.eventLoopGroup = eventLoopGroup
    self.channel = channel
    self.client = client
    self.sharedSecret = sharedSecret
  }

  public func disconnect() throws {
    try channel?.close().wait()
    try eventLoopGroup?.syncShutdownGracefully()
    client = nil
    channel = nil
    eventLoopGroup = nil
    sharedSecret = nil
  }

  deinit {
    guard let eventLoopGroup = eventLoopGroup else { return }
    // Do async shutdown in deinit, make sure we don't hold any of self reference.
    let _ = channel?.close().always { _ in
      eventLoopGroup.shutdownGracefully { _ in }
    }
  }

  public struct LabHours {
    public var community: Int
    public var plus: Int
    public var expireAt: Date
    public init(community: Int, plus: Int, expireAt: Date) {
      self.community = community
      self.plus = plus
      self.expireAt = expireAt
    }
  }

  public func hours(callback: @escaping (LabHours?) -> Void) {
    guard let client = client else {
      callback(nil)
      return
    }
    let request = HoursRequest.with { _ in }

    let callOptions = CallOptions(
      messageEncoding: .enabled(.responsesOnly(decompressionLimit: .ratio(100))))

    let _ = client.hours(request, callOptions: callOptions).response.always {
      switch $0 {
      case .success(let result):
        if result.hasThresholds {
          let thresholds = result.thresholds
          callback(
            LabHours(
              community: Int(thresholds.community), plus: Int(thresholds.plus),
              expireAt: Date(timeIntervalSince1970: TimeInterval(thresholds.expireAt))))
        } else {
          callback(nil)
        }
      case .failure(_):
        callback(nil)
      }
    }
  }

  public func echo(
    callback: @escaping (
      Bool, Bool,
      (
        files: [String], models: [ModelZoo.Specification], LoRAs: [LoRAZoo.Specification],
        controlNets: [ControlNetZoo.Specification],
        textualInversions: [TextualInversionZoo.Specification]
      ),
      LabHours?, UInt64
    ) -> Void
  ) {
    guard let client = client else {
      callback(
        false, false, (files: [], models: [], LoRAs: [], controlNets: [], textualInversions: []),
        nil, 0)
      return
    }

    let request = EchoRequest.with {
      $0.name = deviceName ?? ""
      if let sharedSecret = sharedSecret {
        $0.sharedSecret = sharedSecret
      }
    }

    let callOptions = CallOptions(
      messageEncoding: .enabled(.responsesOnly(decompressionLimit: .ratio(100))))

    let _ = client.echo(request, callOptions: callOptions).response.always {
      switch $0 {
      case .success(let result):
        let jsonDecoder = JSONDecoder()
        jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
        let models =
          (try? jsonDecoder.decode(
            [FailableDecodable<ModelZoo.Specification>].self, from: result.override.models
          ).compactMap({ $0.value })) ?? []
        let loras =
          (try? jsonDecoder.decode(
            [FailableDecodable<LoRAZoo.Specification>].self, from: result.override.loras
          ).compactMap({ $0.value })) ?? []
        let controlNets =
          (try? jsonDecoder.decode(
            [FailableDecodable<ControlNetZoo.Specification>].self, from: result.override.controlNets
          ).compactMap({ $0.value })) ?? []
        let textualInversions =
          (try? jsonDecoder.decode(
            [FailableDecodable<TextualInversionZoo.Specification>].self,
            from: result.override.textualInversions
          ).compactMap({ $0.value })) ?? []
        let labHours: LabHours? = {
          guard result.hasThresholds else { return nil }
          return LabHours(
            community: Int(result.thresholds.community), plus: Int(result.thresholds.plus),
            expireAt: Date(timeIntervalSince1970: TimeInterval(result.thresholds.expireAt)))
        }()
        callback(
          true, !result.sharedSecretMissing,
          (
            files: result.files, models: models, LoRAs: loras, controlNets: controlNets,
            textualInversions: textualInversions
          ), labHours, result.serverIdentifier)
      case .failure(_):
        callback(
          false, false, (files: [], models: [], LoRAs: [], controlNets: [], textualInversions: []),
          nil, 0)
      }
    }
  }

  public typealias FileExistsCall = UnaryCall<FileListRequest, FileExistenceResponse>

  public func filesExists(
    files: [String], filesToMatch: [String],
    callback: @escaping (Bool, [(String, Bool, Data)]) -> Void
  ) -> UnaryCall<FileListRequest, FileExistenceResponse>? {
    guard let client = client else {
      callback(false, [])
      return nil
    }

    let request = FileListRequest.with {
      $0.files = files
      $0.filesWithHash = filesToMatch
      if let sharedSecret = sharedSecret {
        $0.sharedSecret = sharedSecret
      }
    }
    let call = client.filesExist(request)
    let _ = call.response.always { result in
      switch result {
      case .success(let response):
        let payload = zip(response.files, response.existences).enumerated().map {
          ($1.0, $1.1, $0 < response.hashes.count ? response.hashes[$0] : Data())
        }
        callback(true, payload)
      case .failure(_):
        callback(false, [])
      }
    }
    return call
  }
}
