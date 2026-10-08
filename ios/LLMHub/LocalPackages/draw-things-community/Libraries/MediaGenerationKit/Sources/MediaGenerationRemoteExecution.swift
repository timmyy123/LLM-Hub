import DataModels
import Diffusion
import Foundation
import GRPC
import GRPCImageServiceModels
import GRPCServer
import ImageGenerator
import ModelZoo
import NNC
import RemoteImageGenerator

/// Configuration for remote image generation execution.
internal struct MediaGenerationRemoteConfiguration {
  public let serverURL: String
  public let port: Int
  public let useTLS: Bool
  public let authentication: MediaGenerationRemoteAuthenticationMode
  public let deviceName: String

  public init(
    serverURL: String,
    port: Int = 7859,
    useTLS: Bool = true,
    authentication: MediaGenerationRemoteAuthenticationMode = .none,
    deviceName: String = "MediaGenerationKit"
  ) {
    self.serverURL = serverURL
    self.port = port
    self.useTLS = useTLS
    self.authentication = authentication
    self.deviceName = deviceName
  }
}

internal final class MediaGenerationRemoteExecutor: @unchecked Sendable {
  private let authenticationMode: MediaGenerationRemoteAuthenticationMode
  private let cloudAuthenticator: CloudAuthenticator?
  private let queue: DispatchQueue
  private var remoteGeneratorInstance: RemoteImageGenerator?

  private init(
    authenticationMode: MediaGenerationRemoteAuthenticationMode,
    cloudAuthenticator: CloudAuthenticator?,
    queue: DispatchQueue
  ) {
    self.authenticationMode = authenticationMode
    self.cloudAuthenticator = cloudAuthenticator
    self.queue = queue
  }

  static func create(configuration: MediaGenerationRemoteConfiguration) async throws
    -> MediaGenerationRemoteExecutor
  {
    let cloudAuthenticator: CloudAuthenticator?
    switch configuration.authentication {
    case .cloudCompute(let apiKey, let baseURL, _):
      cloudAuthenticator = CloudAuthenticatorRegistry.shared.authenticator(
        apiKey: apiKey,
        baseURL: baseURL
      )
    case .none, .sharedSecret:
      cloudAuthenticator = nil
    }
    let executor = MediaGenerationRemoteExecutor(
      authenticationMode: configuration.authentication,
      cloudAuthenticator: cloudAuthenticator,
      queue: DispatchQueue(label: "com.drawthings.mediagenerationkit.remote", qos: .userInteractive)
    )
    try await executor.configureAsync(configuration)
    return executor
  }

  func prepareForGeneration() async throws {
    guard case .cloudCompute(_, let baseURL, let appCheck) = authenticationMode,
      let cloudAuthenticator
    else {
      return
    }
    let shortTermToken = try await cloudAuthenticator.shortTermToken(appCheck: appCheck)
    _ = await MediaGenerationCloudAuthentication.prefetchPaygEnabled(
      shortTermToken: shortTermToken,
      baseURL: baseURL
    )
  }

  func generate(
    prompt: String,
    negativePrompt: String = "",
    configuration: GenerationConfiguration,
    image: Data? = nil,
    mask: Data? = nil,
    hints: [MediaGenerationExecutionHint] = [],
    cancellationBridge: MediaGenerationCancellationBridge?,
    uploadProgress: ((Int, Int) -> Void)? = nil,
    downloadProgress: ((Int, Int) -> Void)? = nil,
    feedback: @escaping (ImageGeneratorSignpost, Tensor<FloatType>?) -> Bool,
    completion: @escaping (Result<[Tensor<FloatType>], MediaGenerationKitError>) -> Void
  ) {
    let generator: RemoteImageGenerator
    do {
      generator = try configuredGeneratorForExecution(
        requestedModel: configuration.model,
        uploadProgress: uploadProgress,
        downloadProgress: downloadProgress
      )
    } catch let error as MediaGenerationKitError {
      completion(.failure(error))
      return
    } catch {
      completion(.failure(.generationFailed(error.localizedDescription)))
      return
    }

    MediaGenerationExecutionUtilities.generate(
      on: queue,
      generator: generator,
      prompt: prompt,
      negativePrompt: negativePrompt,
      configuration: configuration,
      image: image,
      mask: mask,
      hints: hints,
      cancellationBridge: cancellationBridge,
      feedback: feedback,
      completion: completion
    )
  }

  private func configuredGeneratorForExecution(
    requestedModel: String?,
    uploadProgress: ((Int, Int) -> Void)?,
    downloadProgress: ((Int, Int) -> Void)?
  ) throws -> RemoteImageGenerator {
    guard let remoteGeneratorInstance else {
      throw MediaGenerationKitError.remoteNotConfigured
    }

    if case .cloudCompute = authenticationMode, let requestedModel, let cloudAuthenticator
    {
      let remoteModels = cloudAuthenticator.remoteModels()
      if !remoteModels.contains(requestedModel) {
        throw MediaGenerationKitError.modelNotFoundOnRemote(requestedModel)
      }
    }

    var generator = remoteGeneratorInstance
    generator.transferDataCallback = RemoteImageGenerator.TransferDataCallback(
      beginUpload: { totalBytes in
        uploadProgress?(0, totalBytes)
      },
      beginDownload: { totalBytes in
        downloadProgress?(0, totalBytes)
      },
      remoteDownloads: { bytesReceived, bytesExpected, item, itemsExpected in
        // Remote model download progress
      }
    )
    self.remoteGeneratorInstance = generator
    return generator
  }

  private func configureAsync(_ config: MediaGenerationRemoteConfiguration) async throws {
    try? disconnect()

    #if os(macOS)
      let deviceType: ImageGeneratorDeviceType = .laptop
    #else
      let deviceType: ImageGeneratorDeviceType = .phone
    #endif

    let sharedSecret: String?
    switch config.authentication {
    case .none:
      sharedSecret = nil
    case .sharedSecret(let secret):
      sharedSecret = secret
    case .cloudCompute:
      sharedSecret = nil
    }

    let client = ImageGenerationClientWrapper(deviceName: config.deviceName)

    do {
      try client.connect(
        host: config.serverURL,
        port: config.port,
        TLS: config.useTLS,
        hostnameVerification: config.useTLS,
        sharedSecret: sharedSecret
      )
    } catch {
      throw MediaGenerationKitError.generationFailed("connect failed: \(error)")
    }

    let handshake: RemoteHandshake
    do {
      handshake = try await performHandshake(using: client, timeout: 5)
    } catch {
      try? client.disconnect()
      throw error
    }
    cloudAuthenticator?.updateRemoteModelsFromHandshake(handshake.discoveredRemoteModels)

    let authHandler = createAuthenticationHandler(from: config.authentication)
    remoteGeneratorInstance = RemoteImageGenerator(
      name: config.deviceName,
      deviceType: deviceType,
      client: client,
      serverIdentifier: handshake.serverIdentifier,
      authenticationHandler: authHandler,
      requestExceedLimitHandler: nil
    )
  }

  private func disconnect() throws {
    try remoteGeneratorInstance?.client.disconnect()
    remoteGeneratorInstance = nil
  }

  private func createAuthenticationHandler(
    from mode: MediaGenerationRemoteAuthenticationMode
  ) -> (
    (Bool, Data, GenerationConfiguration, Bool, Int, (@escaping () -> Void) -> Void) -> String?
  )? {
    switch mode {
    case .none, .sharedSecret:
      return nil
    case .cloudCompute(let apiKey, let baseURL, let appCheck):
      let authenticator =
        cloudAuthenticator
        ?? CloudAuthenticatorRegistry.shared.authenticator(
          apiKey: apiKey,
          baseURL: baseURL
        )
      return { _, encodedBlob, configuration, hasImage, shuffleCount, cancellation in
        let shortTermToken: String
        switch self.blockingShortTermToken(authenticator: authenticator, appCheck: appCheck) {
        case .success(let token):
          shortTermToken = token
        case .failure(let error):
          print("[CloudAuth] Failed to get short-term token: \(error.localizedDescription)")
          return nil
        }

        let estimatedComputeUnits = ComputeUnits.from(
          configuration,
          hasImage: hasImage,
          shuffleCount: shuffleCount
        ).map(Double.init)

        return MediaGenerationCloudAuthentication.authenticate(
          shortTermToken: shortTermToken,
          encodedBlob: encodedBlob.base64EncodedString(),
          fromBridge: true,
          estimatedComputeUnits: estimatedComputeUnits,
          baseURL: baseURL,
          timeout: 30,
          cancellation: cancellation,
          onFailure: { reason in
            print("[CloudAuth] /authenticate failed: \(reason)")
          }
        )
      }
    }
  }

  private func blockingShortTermToken(
    authenticator: CloudAuthenticator,
    appCheck: AppCheckConfiguration,
    timeout: TimeInterval = 30
  ) -> Result<String, Error> {
    let semaphore = DispatchSemaphore(value: 0)
    var tokenResult: Result<String, Error> = .failure(MediaGenerationKitError.notConfigured)
    authenticator.getShortTermToken(appCheck: appCheck) {
      tokenResult = $0
      semaphore.signal()
    }
    let waitResult = semaphore.wait(timeout: .now() + timeout)
    guard waitResult != .timedOut else {
      return .failure(MediaGenerationKitError.generationFailed("authentication timed out"))
    }
    return tokenResult
  }

  private struct RemoteHandshake {
    let serverIdentifier: UInt64
    let discoveredRemoteModels: Set<String>
  }

  private func performHandshake(
    using client: ImageGenerationClientWrapper,
    timeout: TimeInterval
  ) async throws -> RemoteHandshake {
    let bridge = MediaGenerationAsyncResultBridge<RemoteHandshake>()
    let timeoutTask = Task {
      try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
      bridge.resume(
        throwing: MediaGenerationKitError.generationFailed("echo timed out (\(Int(timeout))s)"))
    }

    defer {
      timeoutTask.cancel()
    }

    return try await withTaskCancellationHandler(operation: {
      try await withCheckedThrowingContinuation { continuation in
        guard bridge.install(continuation) else { return }
        client.echo { success, authenticated, resources, labHours, serverIdentifier in
          guard success else {
            bridge.resume(
              throwing: MediaGenerationKitError.generationFailed("echo returned success=false"))
            return
          }
          var discoveredRemoteModels = Set<String>()
          for model in resources.models {
            discoveredRemoteModels.insert(model.file)
          }
          for file in resources.files {
            discoveredRemoteModels.insert(file)
          }
          bridge.resume(
            returning: RemoteHandshake(
              serverIdentifier: serverIdentifier,
              discoveredRemoteModels: discoveredRemoteModels
            )
          )
        }
      }
    }, onCancel: {
      timeoutTask.cancel()
      bridge.cancel()
      try? client.disconnect()
    })
  }
}
