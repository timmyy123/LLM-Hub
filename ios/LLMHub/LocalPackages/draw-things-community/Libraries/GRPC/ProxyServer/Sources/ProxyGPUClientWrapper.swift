import BinaryResources
import Foundation
import GRPC
import GRPCImageServiceModels
import Logging
import NIO
import NIOSSL

public final class ProxyGPUClientWrapper {
  public private(set) var deviceName: String? = nil
  private var eventLoopGroup: EventLoopGroup? = nil
  private var channel: GRPCChannel? = nil
  public private(set) var client: ImageGenerationServiceNIOClient? = nil
  private let logger = Logger(label: "com.draw-things.image-generation-proxy-service")
  public init(deviceName: String? = nil) {
    self.deviceName = deviceName
  }

  public func connect(host: String, port: Int) throws {
    try? eventLoopGroup?.syncShutdownGracefully()
    let eventLoopGroup = PlatformSupport.makeEventLoopGroup(loopCount: 1)
    let transportSecurity: GRPCChannelPool.Configuration.TransportSecurity = .plaintext

    var configuration = GRPCChannelPool.Configuration.with(
      target: .host(host, port: port),
      transportSecurity: transportSecurity,
      eventLoopGroup: eventLoopGroup)
    configuration.maximumReceiveMessageLength = 1024 * 1024 * 1024
    let channel = try GRPCChannelPool.with(configuration: configuration)
    let client = ImageGenerationServiceNIOClient(channel: channel)
    self.eventLoopGroup = eventLoopGroup
    self.channel = channel
    self.client = client
  }

  public func disconnect() throws {
    try channel?.close().wait()
    try eventLoopGroup?.syncShutdownGracefully()
    client = nil
    channel = nil
    eventLoopGroup = nil
  }

  deinit {
    guard let eventLoopGroup = eventLoopGroup else { return }
    // Do async shutdown in deinit, make sure we don't hold any of self reference.
    let _ = channel?.close().always { _ in
      eventLoopGroup.shutdownGracefully { _ in }
    }
  }

  public func echo() async -> (Bool, [String]) {
    guard let client = client else {
      return (false, [])
    }

    var request = EchoRequest()
    let name = deviceName ?? ""
    request.name = "Proxy Server connect \(name)"
    let callOptions = CallOptions(
      timeLimit: .timeout(.seconds(Int64(300)))
    )
    do {
      let result = try await client.echo(request, callOptions: callOptions).response.get()
      return (true, result.files)
    } catch {
      return (false, [])
    }
  }

}
