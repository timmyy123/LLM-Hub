import BinaryResources
import Crypto
import DataModels
import Foundation
import GRPC
import GRPCControlPanelModels
import GRPCImageServiceModels
import Logging
import ModelZoo
import NIO
import NIOHPACK
import NIOHTTP2
import NIOSSL

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

public enum ProxyTaskPriority: Sendable {
  case real
  case high
  case low
  case background
}

final class WorkTaskResponseWriter {
  private let context: StreamingResponseCallContext<ImageGenerationResponse>
  private let eventLoop: EventLoop
  private var writeTail: EventLoopFuture<Void>
  private var finished = false
  private var firstFailure: Error?

  init(context: StreamingResponseCallContext<ImageGenerationResponse>) {
    self.context = context
    eventLoop = context.eventLoop
    writeTail = eventLoop.makeSucceededFuture(())
  }

  func send(_ response: ImageGenerationResponse) -> EventLoopFuture<Void> {
    let promise = eventLoop.makePromise(of: Void.self)
    eventLoop.execute {
      if let firstFailure = self.firstFailure {
        promise.fail(firstFailure)
        return
      }
      guard !self.finished else {
        promise.fail(GRPCStatus(code: .cancelled, message: "response stream is finishing"))
        return
      }
      let responseWritten = self.writeTail.flatMap {
        self.context.sendResponse(response)
      }
      self.writeTail = responseWritten
      responseWritten.whenFailure { error in
        if self.firstFailure == nil {
          self.firstFailure = error
        }
      }
      responseWritten.cascade(to: promise)
    }
    return promise.futureResult
  }

  func waitForPendingWrites() -> EventLoopFuture<Void> {
    let promise = eventLoop.makePromise(of: Void.self)
    eventLoop.execute {
      self.finished = true
      self.writeTail.cascade(to: promise)
    }
    return promise.futureResult
  }
}

struct WorkTask {
  var priority: ProxyTaskPriority
  var request: ImageGenerationRequest
  var context: StreamingResponseCallContext<ImageGenerationResponse>
  var responseWriter: WorkTaskResponseWriter
  var promise: EventLoopPromise<GRPCStatus>
  var heartbeat: Task<Void, Error>
  var creationTimestamp: Date
  var model: String
  var payload: JWTPayload
}

public struct Worker {
  var id: String
  var primaryPriority: ProxyTaskPriority
  public let client: ProxyGPUClientWrapper
  private let logger = Logger(label: "com.draw-things.image-generation-proxy-service")
  enum WorkerError: Error {
    case invalidNioClient
  }
  public init(
    id: String, client: ProxyGPUClientWrapper, primaryPriority: ProxyTaskPriority
  ) {
    self.id = id
    self.client = client
    self.primaryPriority = primaryPriority
  }
}

extension Worker {
  func executeTask(
    _ task: WorkTask, proxyMessageSigner: ProxyMessageSigner, throttleQueueTimeoutSeconds: Int
  ) async throws {
    logger.info(
      "Worker \(id) primaryPriority:\(primaryPriority) starting task  (Priority: \(task.priority))"
    )
    let taskQueueingTimeMs = Date().timeIntervalSince(task.creationTimestamp) * 1000
    logger.info(
      "Task queueing time: \(taskQueueingTimeMs)ms, (Priority: \(task.priority)), userId: \(task.payload.userId as Any), generation id: \(task.payload.generationId as Any)"
    )
    defer { task.heartbeat.cancel() }

    // Check if task from throttle queue is over 1 hour old, reject if so (except boost tasks)
    if task.priority == .background && task.payload.consumableType != .boost {
      let taskAge = Date().timeIntervalSince(task.creationTimestamp)
      if taskAge > Double(throttleQueueTimeoutSeconds) {
        let errorMessage = "Task rejected: enqueue time exceeded 1 hour (\(Int(taskAge))s)"
        logger.info(
          "user: \(task.payload.userId as Any), \(errorMessage) (Priority: \(task.priority))")
        // intentionally abort task from throttle queue over 1 hour
        let status = GRPCStatus(code: .aborted, message: errorMessage)
        do {
          try await task.responseWriter.waitForPendingWrites().get()
          task.promise.succeed(status)
        } catch {
          task.promise.fail(error)
          throw error
        }
        return
      }
    }

    do {
      var call: ServerStreamingCall<ImageGenerationRequest, ImageGenerationResponse>? = nil
      guard let client = client.client else {
        logger.error("Worker \(id) task failed: invalid NIO client")
        throw WorkerError.invalidNioClient
      }
      let logger = logger
      var numberOfImages = 0
      let taskExecuteStartTimestamp = Date()
      let workerId = id
      let callInstance = client.generateImage(task.request) { response in
        if !response.generatedImages.isEmpty {
          numberOfImages += response.generatedImages.count
        }
        task.responseWriter.send(response).whenComplete { result in
          switch result {
          case .success:
            logger.debug("forward response: \(response)")
          case .failure(let error):
            logger.error("Worker: \(workerId), forward response error \(error)")
            call?.cancel(promise: nil)
            task.promise.fail(error)
          }
        }
      }

      call = callInstance
      task.context.closeFuture.whenComplete { _ in
        logger.info(
          "Worker \(workerId) downstream call (proxy<->client) stream closed, model:\(task.model), userId:\(task.payload.userId as Any), generationId:\(task.payload.generationId as Any), priority:\(task.priority)"
        )
        callInstance.cancel(promise: nil)
      }

      let userIdDescription = "\(task.payload.userId as Any)"
      let generationIdDescription = "\(task.payload.generationId as Any)"
      logger.info(
        "Worker \(id) waiting for upstream call (proxy<->GPU) status, model:\(task.model), userId:\(userIdDescription), generationId:\(generationIdDescription), priority:\(task.priority)"
      )
      let status: GRPCStatus
      do {
        status = try await callInstance.status.get()
        logger.info(
          "Worker \(id) upstream call (proxy<->GPU) status returned, model:\(task.model), userId:\(userIdDescription), generationId:\(generationIdDescription), status:\(status.code), message:\(status.message ?? ""), priority:\(task.priority)"
        )
      } catch {
        logger.error(
          "Worker \(id) upstream call (proxy<->GPU) status failed, model:\(task.model), userId:\(userIdDescription), generationId:\(generationIdDescription), error:\(error), priority:\(task.priority)"
        )
        throw error
      }
      if numberOfImages > 0 {
        let totalTimeMs = Date().timeIntervalSince(task.creationTimestamp) * 1000
        let totalExecutionTimeMs = Date().timeIntervalSince(taskExecuteStartTimestamp) * 1000
        logger.info(
          "Task total time: \(totalTimeMs)ms, Task execution time: \(totalExecutionTimeMs)ms, (Priority: \(task.priority))"
        )
        logger.info(
          "Succeed: {\"model\": \"\(task.model)\", \"userid\": \"\(userIdDescription)\",  \"generationId\": \"\(generationIdDescription)\", \"images\":\(numberOfImages)}"
        )
      }
      let isTaskSuccessful = (status.code == .ok) && (numberOfImages > 0)
      if status.code != .ok {
        logger.error(
          "Worker \(id) completed with non-OK status, model:\(task.model), userId:\(userIdDescription), generationId:\(generationIdDescription), status:\(status.code), message:\(status.message ?? ""), images:\(numberOfImages), priority:\(task.priority)"
        )
      } else if numberOfImages == 0 {
        logger.error(
          "Worker \(id) returned OK with zero generated image responses, model:\(task.model), userId:\(userIdDescription), generationId:\(generationIdDescription), priority:\(task.priority)"
        )
      }

      // callInstance.status only means the upstream call (proxy<->GPU) has completed.
      // Response callbacks may have already enqueued downstream writes (proxy<->client)
      // that are still pending. Drain those forwarded responses before completing the
      // provider future; otherwise grpc-swift may send downstream status/trailers before
      // every forwarded response has been flushed.

      try await task.responseWriter.waitForPendingWrites().get()
      if task.payload.consumableType == .boost, let amount = task.payload.amount,
        let generationId = task.payload.generationId
      {
        await proxyMessageSigner.completeBoost(
          action: isTaskSuccessful ? .complete : .cancel, generationId: generationId,
          amount: amount,
          logger: logger)
      } else if task.payload.consumableType == .payg, let amount = task.payload.amount,
        let generationId = task.payload.generationId
      {
        await proxyMessageSigner.completePayg(
          action: isTaskSuccessful ? .complete : .cancel, generationId: generationId,
          amount: amount,
          logger: logger)
      }
      logger.info(
        "Worker \(id) completing downstream call (proxy<->client) status, model:\(task.model), userId:\(userIdDescription), generationId:\(generationIdDescription), status:\(status.code), message:\(status.message ?? ""), priority:\(task.priority)"
      )
      task.promise.succeed(status)

      if isTaskSuccessful {
        logger.info(
          "Worker \(id) completed, generationId: \(generationIdDescription), successfully (Priority: \(task.priority))"
        )
      } else {
        logger.info(
          "Worker \(id) completed without successful final image, generationId: \(generationIdDescription), status:\(status.code), images:\(numberOfImages), (Priority: \(task.priority))"
        )
      }

    } catch {
      if task.payload.consumableType == .boost, let amount = task.payload.amount,
        let generationId = task.payload.generationId
      {
        await proxyMessageSigner.completeBoost(
          action: .cancel, generationId: generationId, amount: amount,
          logger: logger)
      } else if task.payload.consumableType == .payg, let amount = task.payload.amount,
        let generationId = task.payload.generationId
      {
        await proxyMessageSigner.completePayg(
          action: .cancel, generationId: generationId, amount: amount,
          logger: logger)
      }
      logger.error("Worker \(id) task failed with error: \(error) (Priority: \(task.priority))")
      task.promise.fail(error)
      throw error
    }
  }

}

actor TaskQueue {
  private struct DeathPoolEntry {
    var worker: Worker
    var addedAt: Date
  }

  private var highPriorityTasks: [WorkTask] = []
  private var lowPriorityTasks: [WorkTask] = []
  private var realPriorityTasks: [WorkTask] = []
  private var backgroundPriorityTasks: [WorkTask] = []
  private var pendingRemoveWorkerId = Set<String>()
  private var workers: [String: Worker]
  private var deathPool: [String: DeathPoolEntry] = [:]
  private var busyWorkerIDs: Set<String> = []
  private var workerEchoRejoinTimestamps = [String: [Date]]()
  private var workerEchoRejoinWindow: TimeInterval = 1800
  private var workerEchoRejoinLimit = 100
  private let logger: Logger
  var workerIds: [String] {
    return Array(workers.keys)
  }

  // Shared availability stream
  private let workerAvailabilityStream: AsyncStream<Worker>
  private let availabilityContinuation: AsyncStream<Worker>.Continuation

  init(workers: [Worker], logger: Logger) {
    self.logger = logger
    self.workers = Dictionary(uniqueKeysWithValues: workers.map { ($0.id, $0) })

    (workerAvailabilityStream, availabilityContinuation) = AsyncStream.makeStream(of: Worker.self)
    for worker in workers {
      availabilityContinuation.yield(worker)
    }
  }

  var availableWorkerCount: Int {
    return workers.count - busyWorkerIDs.count
  }

  func nextWorker() async -> Worker? {
    for await worker in workerAvailabilityStream {
      if workers[worker.id] != nil {
        busyWorkerIDs.insert(worker.id)
        return worker
      } else {
        logger.info("skip removed worker:\(worker) from workerAvailabilityStream")
      }
    }
    return nil
  }

  func nextTaskForWorker(
    _ worker: Worker, highThreshold: Int, communityThreshold: Int, taskLoopBreakoutSeconds: Int
  ) async
    -> WorkTask?
  {
    let startTime = Date()
    let timeout: TimeInterval = TimeInterval(taskLoopBreakoutSeconds)
    // When free workers are very low, only process real and high priority tasks
    while availableWorkerCount < highThreshold && Date().timeIntervalSince(startTime) < timeout {
      if let task = realPriorityTasks.first {
        realPriorityTasks.removeFirst()
        return task
      } else if let task = highPriorityTasks.first {
        highPriorityTasks.removeFirst()
        return task
      } else if availableWorkerCount >= communityThreshold {
        if let task = lowPriorityTasks.first {
          lowPriorityTasks.removeFirst()
          return task
        }
      }
      do {
        try await Task.sleep(for: .milliseconds(5))
      } catch {
        logger.error("Task.sleep failed with error: \(error)")
        continue
      }
    }

    if Date().timeIntervalSince(startTime) >= timeout {
      logger.info(
        "Throttling loop timed out after 30s. availableWorkerCount:\(availableWorkerCount), communityThreshold:\(communityThreshold), availableWorkerCount:\(availableWorkerCount)."
      )
    }

    // Check worker's primary priority queue first
    switch worker.primaryPriority {
    case .real:
      if let task = realPriorityTasks.first {
        realPriorityTasks.removeFirst()
        return task
      }
    case .high:
      if let task = highPriorityTasks.first {
        highPriorityTasks.removeFirst()
        return task
      }
    case .low:
      if let task = lowPriorityTasks.first {
        lowPriorityTasks.removeFirst()
        return task
      }
    case .background:
      if let task = backgroundPriorityTasks.first {
        backgroundPriorityTasks.removeFirst()
        return task
      }
    }

    // If primary queue is empty, check other queues in priority order
    if let task = realPriorityTasks.first {
      realPriorityTasks.removeFirst()
      return task
    }
    if let task = highPriorityTasks.first {
      highPriorityTasks.removeFirst()
      return task
    }
    if let task = lowPriorityTasks.first {
      lowPriorityTasks.removeFirst()
      return task
    }
    if let task = backgroundPriorityTasks.first {
      backgroundPriorityTasks.removeFirst()
      return task
    }

    return nil
  }

  func addTask(_ task: WorkTask) {
    switch task.priority {
    case .real:
      logger.info("realPriorityTasks append task \(task.priority)")
      realPriorityTasks.append(task)
    case .high:
      logger.info("highPriorityTasks append task \(task.priority)")
      highPriorityTasks.append(task)
    case .low:
      logger.info("lowPriorityTasks append task \(task.priority)")
      lowPriorityTasks.append(task)
    case .background:
      logger.info("backgroundPriorityTasks append task \(task.priority)")
      backgroundPriorityTasks.append(task)
    }
  }

  func returnWorker(_ worker: Worker) async {
    guard workers[worker.id] != nil else {
      logger.error("worker:\(worker) is removed, can not be added to worker stream")
      busyWorkerIDs.remove(worker.id)
      return
    }
    logger.info("add worker:\(worker) back to worker stream")
    busyWorkerIDs.remove(worker.id)
    availabilityContinuation.yield(worker)
  }

  private func isWorkerOnHold(_ workerId: String, afterRecordingRejoin: Bool = false) -> Bool {
    let now = Date()
    let cutoff = now.addingTimeInterval(-workerEchoRejoinWindow)
    var timestamps = workerEchoRejoinTimestamps[workerId, default: []].filter { $0 >= cutoff }
    if afterRecordingRejoin {
      timestamps.append(now)
    }
    if timestamps.isEmpty {
      workerEchoRejoinTimestamps.removeValue(forKey: workerId)
    } else {
      workerEchoRejoinTimestamps[workerId] = timestamps
    }
    return timestamps.count > workerEchoRejoinLimit
  }

  private func currentOnHoldWorkerIds() -> [String] {
    let cutoff = Date().addingTimeInterval(-workerEchoRejoinWindow)
    var onHoldWorkerIds = [String]()
    for (workerId, timestamps) in workerEchoRejoinTimestamps {
      let recentTimestamps = timestamps.filter { $0 >= cutoff }
      if recentTimestamps.isEmpty {
        workerEchoRejoinTimestamps.removeValue(forKey: workerId)
      } else {
        workerEchoRejoinTimestamps[workerId] = recentTimestamps
      }
      if recentTimestamps.count > workerEchoRejoinLimit {
        onHoldWorkerIds.append(workerId)
      }
    }
    return onHoldWorkerIds.sorted()
  }

  func clearWorkerEchoRejoinHold(_ workerId: String) async -> (
    cleared: Bool, onHoldWorkerIds: [String]
  ) {
    let cleared = workerEchoRejoinTimestamps.removeValue(forKey: workerId) != nil
    let onHoldWorkerIds = currentOnHoldWorkerIds()
    logger.info(
      "clear echo rejoin hold for worker \(workerId), cleared:\(cleared), current on hold workers:\(onHoldWorkerIds)"
    )
    return (cleared, onHoldWorkerIds)
  }

  func updateWorkerEchoRejoinPolicy(windowSeconds: Int?, limit: Int?) async -> (
    windowSeconds: Int, limit: Int
  ) {
    if let windowSeconds = windowSeconds, windowSeconds > 0 {
      workerEchoRejoinWindow = TimeInterval(windowSeconds)
    }
    if let limit = limit, limit >= 0 {
      workerEchoRejoinLimit = limit
    }
    _ = currentOnHoldWorkerIds()
    let currentWindowSeconds = Int(workerEchoRejoinWindow)
    logger.info(
      "update echo rejoin policy, worker_echo_rejoin_window_seconds:\(currentWindowSeconds), worker_echo_rejoin_limit:\(workerEchoRejoinLimit)"
    )
    return (currentWindowSeconds, workerEchoRejoinLimit)
  }

  func addWorker(_ worker: Worker) async -> Bool {
    guard worker.client.client != nil else {
      logger.error(
        "can add worker:\(worker) to worker TaskQueue with invalid nioclient connection")
      return false
    }
    guard !isWorkerOnHold(worker.id, afterRecordingRejoin: true) else {
      if let existingWorker = workers.removeValue(forKey: worker.id),
        existingWorker.client !== worker.client
      {
        try? existingWorker.client.disconnect()
      }
      busyWorkerIDs.remove(worker.id)
      if let entry = deathPool.removeValue(forKey: worker.id),
        entry.worker.client !== worker.client
      {
        try? entry.worker.client.disconnect()
      }
      logger.info(
        "on hold worker \(worker.id), current on hold workers:\(currentOnHoldWorkerIds())")
      return false
    }
    let alreadyExists = workers[worker.id] != nil
    workers[worker.id] = worker
    guard !alreadyExists else {
      logger.info("worker:\(worker) already exists in workers, skip adding")
      return true
    }
    availabilityContinuation.yield(worker)
    logger.info("add worker:\(worker) to worker TaskQueue and stream")
    return true
  }

  func quarantineWorkerById(_ name: String) async {
    guard let worker = workers[name] else {
      logger.error("failed to find worker based on name \(name)")
      return
    }
    if let entry = deathPool[worker.id], entry.worker.client !== worker.client {
      try? entry.worker.client.disconnect()
    }
    workers[worker.id] = nil
    busyWorkerIDs.remove(worker.id)
    deathPool[worker.id] = DeathPoolEntry(worker: worker, addedAt: Date())
    logger.info(
      "move worker:\(worker) from worker TaskQueue to death pool, death pool workers:\(Array(deathPool.keys))"
    )
  }

  func deathPoolWorkers() async -> [Worker] {
    return deathPool.values.map { $0.worker }
  }

  func shouldAttemptDeathPoolWorker(_ workerId: String) async -> Bool {
    return workers[workerId] == nil && deathPool[workerId] != nil
  }

  func restoreWorkerFromDeathPool(_ worker: Worker) async -> Bool {
    guard let deathPoolEntry = deathPool[worker.id] else {
      logger.info("worker:\(worker.id) is no longer in death pool, skip restoring")
      return false
    }
    if let existingWorker = workers[worker.id] {
      if existingWorker.client !== worker.client {
        try? worker.client.disconnect()
      }
      deathPool.removeValue(forKey: worker.id)
      logger.error(
        "worker:\(worker.id) already exists in worker group while in death pool; removed duplicate death pool entry"
      )
      return existingWorker.client === worker.client
    }
    guard worker.client.client != nil else {
      logger.error("worker:\(worker.id) has invalid nioclient connection, keep in death pool")
      return false
    }
    guard !isWorkerOnHold(worker.id) else {
      deathPool.removeValue(forKey: worker.id)
      logger.info(
        "on hold worker \(worker.id), current on hold workers:\(currentOnHoldWorkerIds())")
      return false
    }
    workers[worker.id] = worker
    deathPool.removeValue(forKey: worker.id)
    availabilityContinuation.yield(worker)
    logger.info(
      "restore worker:\(worker) from death pool after \(Date().timeIntervalSince(deathPoolEntry.addedAt))s, death pool workers:\(Array(deathPool.keys))"
    )
    return true
  }

  func removeDeathPoolWorkerById(_ name: String) async {
    guard let entry = deathPool.removeValue(forKey: name) else {
      return
    }
    try? entry.worker.client.disconnect()
    logger.info("remove worker:\(name) from death pool")
  }

  func removeWorkerById(_ name: String) async {
    guard let worker = workers[name] else {
      if let entry = deathPool.removeValue(forKey: name) {
        try? entry.worker.client.disconnect()
        logger.info("remove worker:\(name) from death pool")
      } else {
        logger.error("failed to find worker based on name \(name)")
      }
      return
    }
    try? worker.client.disconnect()
    workers[worker.id] = nil
    busyWorkerIDs.remove(worker.id)
    if let entry = deathPool.removeValue(forKey: worker.id), entry.worker.client !== worker.client {
      try? entry.worker.client.disconnect()
    }
    logger.info("remove worker:\(worker) from worker TaskQueue")
  }

  deinit {
    for worker in workers.values {
      try? worker.client.disconnect()
    }
    for entry in deathPool.values {
      try? entry.worker.client.disconnect()
    }
    availabilityContinuation.finish()
  }
}

public actor ControlConfigs {
  public private(set) var throttlePolicy = [String: Int]()
  public private(set) var publicKeyPEM: String
  public private(set) var modelListPath: String
  public private(set) var computeUnitPerBoost: Int = 60000
  private var nonces = Set<String>()
  private let logger: Logger
  private var nonceSizeLimit: Int
  private var sharedSecret: String?
  private var computeUnitPolicy = [String: Int]()
  private var expirationTimestamp: Date?
  enum ControlConfigsError: Error {
    case updatePublicKeyFailed(message: String)
  }

  init(
    throttlePolicy: [String: Int], publicKeyPEM: String, logger: Logger, modelListPath: String,
    nonceSizeLimit: Int
  ) {
    self.throttlePolicy = throttlePolicy
    self.publicKeyPEM = publicKeyPEM
    self.logger = logger
    self.modelListPath = modelListPath
    self.nonceSizeLimit = nonceSizeLimit
  }

  func addProcessedNonce(_ nonce: String) async {
    guard !nonces.contains(nonce) else { return }
    if nonces.count >= nonceSizeLimit {
      if let randomElement = nonces.randomElement() {
        nonces.remove(randomElement)
      }
    }

    nonces.insert(nonce)
    logger.info("ControlConfigs add processed nonce:\(nonce)")
  }

  func getComputeUnitPolicyAndExpirationTimestamp() async -> ([String: Int], Date?) {
    return (computeUnitPolicy, expirationTimestamp)
  }

  func isUsedNonce(_ nonce: String) async -> Bool {
    nonces.contains(nonce)
  }

  func isSharedSecretValid(_ sharedSecret: String) async -> Bool {
    guard let configuredSecret = self.sharedSecret else {
      return false
    }
    return configuredSecret == sharedSecret
  }

  func updateThrottlePolicy(newPolicies: [String: Int]) async {
    for (key, value) in newPolicies {
      throttlePolicy[key] = value
    }
  }

  func updateComputeUnitPolicyAndExpirationTimestamp(newPolicies: [String: Int], _ timestamp: Date)
    async
  {
    for (key, value) in newPolicies {
      computeUnitPolicy[key] = value
    }
    expirationTimestamp = timestamp
  }

  func updatePublicKeyPEM() async throws {
    guard let url = URL(string: "https://api.drawthings.ai/key") else {
      logger.error("ControlConfigs failed to update Pem, invalid url")
      return
    }
    let (data, response) = try await URLSession.shared.data(from: url)
    if let string = String(data: data, encoding: .utf8) {
      publicKeyPEM = string
      logger.info("ControlConfigs update public Pem as:\(string)")
    } else {
      logger.error("ControlConfigs failed to update Pem, response:\(response))")
    }
  }

  func updateSharedSecret() async -> String {
    func generatePassword() -> String {
      // Avoiding confusing characters like 1, l, I, i, 0, O
      let allowedChars = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
      var password = ""
      // Ensure at least one number and one letter
      let numbers = "23456789"
      let letters = "ABCDEFGHJKMNPQRSTUVWXYZ"
      // Add one random number
      password.append(numbers.randomElement()!)
      // Add one random letter
      password.append(letters.randomElement()!)
      // Fill the remaining 10 characters
      for _ in 0..<10 {
        password.append(allowedChars.randomElement()!)
      }
      // Shuffle the password to randomize position of guaranteed number and letter
      return String(password.shuffled())
    }
    let sharedSecret = generatePassword()
    self.sharedSecret = sharedSecret
    logger.info("ControlConfigs update shared Secret")
    return sharedSecret
  }
}

final class ControlPanelService: ControlPanelServiceProvider {
  var interceptors:
    (any GRPCControlPanelModels.ControlPanelServiceServerInterceptorFactoryProtocol)?
  private let taskQueue: TaskQueue
  private var controlConfigs: ControlConfigs
  private var proxyMessageSigner: ProxyMessageSigner
  private let logger: Logger
  enum ControlPanelError: Error {
    case gpuConnectFailed(message: String)
    case nioClientFailed(message: String)
    case removeGPUFailed(message: String)
  }

  init(
    taskQueue: TaskQueue, controlConfigs: ControlConfigs, logger: Logger,
    proxyMessageSigner: ProxyMessageSigner
  ) {
    self.taskQueue = taskQueue
    self.controlConfigs = controlConfigs
    self.logger = logger
    self.proxyMessageSigner = proxyMessageSigner
  }

  func manageGPUServer(
    request: GRPCControlPanelModels.GPUServerRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.GPUServerResponse> {
    let promise = context.eventLoop.makePromise(of: GRPCControlPanelModels.GPUServerResponse.self)
    Task {
      let gpuServerName = "\(request.serverConfig.address):\(request.serverConfig.port)"
      switch request.operation {
      case .add:
        self.logger.info(
          "Worker connecting server: \(gpuServerName). Worker focus on \(request.serverConfig.isHighPriority ? ProxyTaskPriority.high : ProxyTaskPriority.low) priority Task"
        )
        let client = ProxyGPUClientWrapper(deviceName: gpuServerName)
        do {
          try client.connect(
            host: request.serverConfig.address, port: Int(request.serverConfig.port))
          let result = await client.echo()
          self.logger.info("server: \(gpuServerName). echo \(result)")
          if let _ = client.client, result.0 {
            let worker = Worker(
              id: gpuServerName, client: client,
              primaryPriority: request.serverConfig.isHighPriority ? .high : .low)
            await taskQueue.removeDeathPoolWorkerById(gpuServerName)
            let added = await taskQueue.addWorker(worker)
            let workersId = await taskQueue.workerIds
            let message: String
            if added {
              message =
                "added GPU \(gpuServerName) into workers stream, current workers:\(workersId)"
            } else {
              try? client.disconnect()
              message = "on hold worker \(gpuServerName), current workers:\(workersId)"
            }
            let response = GPUServerResponse.with {
              $0.message = message
            }
            promise.succeed(response)
          } else {
            try? client.disconnect()
            promise.fail(
              ControlPanelError.nioClientFailed(
                message: "fail to create nio client for \(gpuServerName)"))
          }
        } catch (let error) {
          promise.fail(
            ControlPanelError.gpuConnectFailed(
              message: "fail to connect GPU \(gpuServerName) error:\(error)"))
        }
      case .remove:
        await taskQueue.removeWorkerById(gpuServerName)
        let workersId = await taskQueue.workerIds
        let response = GPUServerResponse.with {
          $0.message =
            "remove GPU \(gpuServerName) from taskCoordinator, current workers:\(workersId)"
        }
        promise.succeed(response)
      case .clearEchoHold:
        let result = await taskQueue.clearWorkerEchoRejoinHold(gpuServerName)
        let response = GPUServerResponse.with {
          if result.cleared {
            $0.message =
              "cleared echo hold for GPU \(gpuServerName), current on hold workers:\(result.onHoldWorkerIds)"
          } else {
            $0.message =
              "no echo hold record for GPU \(gpuServerName), current on hold workers:\(result.onHoldWorkerIds)"
          }
        }
        promise.succeed(response)
      case .unspecified, .UNRECOGNIZED(_):
        break
      }

    }

    return promise.futureResult
  }

  func updateModelList(
    request: GRPCControlPanelModels.UpdateModelListRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.UpdateModelListResponse> {
    let promise = context.eventLoop.makePromise(
      of: GRPCControlPanelModels.UpdateModelListResponse.self)
    Task {
      let fileList = request.files.joined(separator: "\n")
      // TODO: The path is problematic.
      let internalFilePath = await controlConfigs.modelListPath
      try? fileList.write(
        to: URL(fileURLWithPath: internalFilePath),
        atomically: true,
        encoding: .utf8)
      self.logger.info(
        "update model list to file: \(internalFilePath) with \(request.files.count) models")

      let response = UpdateModelListResponse.with {
        $0.message = "update model-list file with \(request.files.count) models"
      }
      promise.succeed(response)

    }

    return promise.futureResult
  }

  func updateThrottlingConfig(
    request: GRPCControlPanelModels.ThrottlingRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.ThrottlingResponse> {
    let promise = context.eventLoop.makePromise(of: GRPCControlPanelModels.ThrottlingResponse.self)
    Task {
      await controlConfigs.updateThrottlePolicy(
        newPolicies: request.limitConfig.mapValues { Int($0) })
      let currentThrottlePolicies = await controlConfigs.throttlePolicy
      let echoRejoinPolicy = await taskQueue.updateWorkerEchoRejoinPolicy(
        windowSeconds: currentThrottlePolicies["worker_echo_rejoin_window_seconds"],
        limit: currentThrottlePolicies["worker_echo_rejoin_limit"])
      self.logger.info(
        "Update throttling for \(request.limitConfig), current throttling policies are \(currentThrottlePolicies), echo rejoin policy is \(echoRejoinPolicy)"
      )
      let response = ThrottlingResponse.with {
        $0.message =
          "Update throttling for \(request.limitConfig), current throttling policies are \(currentThrottlePolicies), echo rejoin policy is \(echoRejoinPolicy)"
      }
      promise.succeed(response)
    }

    return promise.futureResult
  }

  func updatePem(
    request: GRPCControlPanelModels.UpdatePemRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.UpdatePemResponse> {
    let promise = context.eventLoop.makePromise(of: GRPCControlPanelModels.UpdatePemResponse.self)
    Task {
      try await controlConfigs.updatePublicKeyPEM()
      let pem = await controlConfigs.publicKeyPEM
      let response = UpdatePemResponse.with {
        $0.message = "Update proxy pem as:\n \(pem)"
      }
      promise.succeed(response)
    }
    return promise.futureResult
  }

  func updateSharedSecret(request: UpdateSharedSecretRequest, context: StatusOnlyCallContext)
    -> EventLoopFuture<UpdateSharedSecretResponse>
  {
    let promise = context.eventLoop.makePromise(
      of: GRPCControlPanelModels.UpdateSharedSecretResponse.self)
    Task {
      let sharedSecret = await controlConfigs.updateSharedSecret()
      let response = UpdateSharedSecretResponse.with {
        $0.message = "Update proxy shared secret as:\n \(sharedSecret)"
      }
      promise.succeed(response)
    }

    return promise.futureResult
  }

  func updatePrivateKey(request: UpdatePrivateKeyRequest, context: StatusOnlyCallContext)
    -> EventLoopFuture<UpdatePrivateKeyResponse>
  {
    let promise = context.eventLoop.makePromise(
      of: GRPCControlPanelModels.UpdatePrivateKeyResponse.self)
    Task {
      await proxyMessageSigner.reloadKeys()
      self.logger.info(
        "regenerate proxy private key pairs"
      )

      let publicKeyPEM = await proxyMessageSigner.getPublicKey()
      let response = UpdatePrivateKeyResponse.with {
        $0.message = "Update proxy private keys, current public key is: \(publicKeyPEM as Any)"
      }
      promise.succeed(response)
    }

    return promise.futureResult
  }

  func updateComputeUnit(
    request: GRPCControlPanelModels.UpdateComputeUnitRequest,
    context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.UpdateComputeUnitResponse> {
    let promise = context.eventLoop.makePromise(
      of: GRPCControlPanelModels.UpdateComputeUnitResponse.self)
    Task {
      // Get current state
      let (currentComputeUnitPolicies, currentExpirationTimestamp) =
        await controlConfigs.getComputeUnitPolicyAndExpirationTimestamp()
      let currentExpirationStr =
        currentExpirationTimestamp.map { "expiration: \($0)" } ?? "no expiration"

      // Validate expiration timestamp first
      let currentTime = Int64(Date().timeIntervalSince1970)

      let logMessage: String
      let response: UpdateComputeUnitResponse
      let newExpirationTimestamp = request.expirationTimestamp
      if newExpirationTimestamp > currentTime {
        self.logger.info(
          "Current compute unit policies: \(currentComputeUnitPolicies), \(currentExpirationStr)")

        // Update with new policies and timestamp
        await controlConfigs.updateComputeUnitPolicyAndExpirationTimestamp(
          newPolicies: request.cuConfig.mapValues { Int($0) },
          Date(timeIntervalSince1970: TimeInterval(newExpirationTimestamp)))

        let (updatedComputeUnitPolicies, updatedExpirationTimestamp) =
          await controlConfigs.getComputeUnitPolicyAndExpirationTimestamp()

        logMessage =
          "Updated compute unit policies to \(updatedComputeUnitPolicies), timestamp: \(updatedExpirationTimestamp as Any)"

        response = UpdateComputeUnitResponse.with {
          $0.message = logMessage
        }
      } else {
        // Invalid timestamp - skip update and report current state
        logMessage =
          "Invalid expiration timestamp \(request.expirationTimestamp) (must be in the future). Current compute unit policies: \(currentComputeUnitPolicies), \(currentExpirationStr)"

        response = UpdateComputeUnitResponse.with {
          $0.message = logMessage
        }
      }

      self.logger.info("\(logMessage)")
      promise.succeed(response)
    }

    return promise.futureResult
  }
}

final class ImageGenerationProxyService: ImageGenerationServiceProvider {
  var interceptors:
    (any GRPCImageServiceModels.ImageGenerationServiceServerInterceptorFactoryProtocol)?

  private let taskQueue: TaskQueue
  private let logger: Logger
  private var controlConfigs: ControlConfigs
  private var healthCheckTask: Task<Void, Never>?
  private var deathPoolRevivalTask: Task<Void, Never>?
  private var proxyMessageSigner: ProxyMessageSigner
  private var workerFailureCounts: [String: Int] = [:]
  private let maxFailureCount: Int
  private let deathPoolRevivalIntervalSeconds = 6 * 60 * 60

  init(
    taskQueue: TaskQueue, controlConfigs: ControlConfigs, logger: Logger, healthCheck: Bool,
    proxyMessageSigner: ProxyMessageSigner, maxFailureCount: Int = 3
  ) {
    self.taskQueue = taskQueue
    self.logger = logger
    self.controlConfigs = controlConfigs
    self.proxyMessageSigner = proxyMessageSigner
    self.maxFailureCount = maxFailureCount
    if healthCheck {
      self.startHealthCheck()
      self.startDeathPoolRevivalTask()
    }
  }

  private func startHealthCheck() {
    healthCheckTask = Task { [weak self] in
      while !Task.isCancelled {
        guard let self = self else { break }
        if let worker = await taskQueue.nextWorker() {
          let (success, _) = await worker.client.echo()
          if success {
            logger.info("Health check passed for worker: \(worker.id)")
            workerFailureCounts[worker.id] = 0
            await taskQueue.returnWorker(worker)
          } else {
            let currentCount = workerFailureCounts[worker.id, default: 0] + 1
            workerFailureCounts[worker.id] = currentCount
            logger.error(
              "Health check failed for worker: \(worker.id) (failure count: \(currentCount)/\(maxFailureCount))"
            )

            if currentCount >= maxFailureCount {
              logger.error("Worker \(worker.id) exceeded max failure count, moving to death pool")
              workerFailureCounts.removeValue(forKey: worker.id)
              await taskQueue.quarantineWorkerById(worker.id)
            } else {
              await taskQueue.returnWorker(worker)
            }
          }
          try? await Task.sleep(for: .seconds(10))  // wait for 10s
        }
      }
    }
  }

  private func startDeathPoolRevivalTask() {
    let intervalSeconds = deathPoolRevivalIntervalSeconds
    deathPoolRevivalTask = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(intervalSeconds))
        guard let self = self, !Task.isCancelled else { break }
        await self.reviveDeathPoolWorkers()
      }
    }
  }

  private func reviveDeathPoolWorkers() async {
    let workers = await taskQueue.deathPoolWorkers()
    guard !workers.isEmpty else {
      logger.info("death pool revive skipped, no workers in death pool")
      return
    }
    logger.info("try revive \(workers.count) worker(s) from death pool")
    for worker in workers {
      guard !Task.isCancelled else { return }
      guard await taskQueue.shouldAttemptDeathPoolWorker(worker.id) else {
        continue
      }
      let (success, _) = await worker.client.echo()
      if success {
        logger.info("death pool worker:\(worker.id) echo succeeded, restoring to worker group")
        let restored = await taskQueue.restoreWorkerFromDeathPool(worker)
        if !restored {
          try? worker.client.disconnect()
        }
      } else {
        logger.error("death pool worker:\(worker.id) echo failed, keep in death pool")
      }
    }
  }

  deinit {
    healthCheckTask?.cancel()
    deathPoolRevivalTask?.cancel()
  }

  func parseBearer(from string: String) -> String? {
    let components = string.trimmingCharacters(in: .whitespaces).split(separator: " ")
    guard components.count == 2,
      components[0].lowercased() == "bearer"
    else {
      return nil
    }
    return String(components[1])
  }

  private func isValidRequest(
    payload: JWTPayload?, encodedBlob: Data, request: ImageGenerationRequest
  ) async -> (Bool, String, String?) {

    let isSharedSecretValid = await controlConfigs.isSharedSecretValid(request.sharedSecret)
    logger.info(
      "Proxy Server enqueue image generating payload:\(payload as Any)"
    )
    if isSharedSecretValid {
      logger.info("Proxy Server SharedSecret is valid, skip requests validation")
      return (true, "", GenerationConfiguration.from(data: request.configuration).model)
    }
    let requestHash = Data(SHA256.hash(data: encodedBlob))
    let checksum = requestHash.map({ String(format: "%02x", $0) }).joined()
    guard let payload = payload, payload.checksum == checksum else {
      logger.info(
        "Proxy Server enqueue image generating request failed, payload.blobSHA:\(payload?.checksum ?? "empty"), request blob:\(checksum) "
      )
      logger.info(
        "Proxy Server enqueue image generating request failed, payload:\(payload as Any)"
      )
      return (false, "Service bear-token signature is failed", nil)
    }
    logger.info("Proxy Server verified request checksum:\(checksum) success")

    guard await !controlConfigs.isUsedNonce(payload.nonce) || isSharedSecretValid else {
      logger.error(
        "Proxy Server image generating request failed, \(payload.nonce) is a used nonce"
      )
      return (false, "used nonce", nil)
    }
    await self.controlConfigs.addProcessedNonce(payload.nonce)
    let throttlePolicies = await controlConfigs.throttlePolicy
    for (key, stat) in payload.stats {
      let effectivePolicy = getEffectiveThrottlePolicy(
        for: key, payload: payload, throttlePolicies: throttlePolicies)

      if let throttlePolicy = effectivePolicy, throttlePolicy < stat {
        logger.error(
          "user \(payload.userId as Any) made \(stat) requests, while policy only allow \(throttlePolicy) for \(key)"
        )
        return (
          false, "user failed to pass throttlePolicy, \(key) in \(throttlePolicy)", nil
        )
      }
    }
    let configuration = GenerationConfiguration.from(data: request.configuration)
    guard let model = configuration.model else {
      return (false, "no valid model name ", nil)
    }
    // decode override models mapping
    let override = request.override
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    let overrideModels =
      (try? jsonDecoder.decode(
        [FailableDecodable<ModelZoo.Specification>].self, from: override.models
      ).compactMap({ $0.value })) ?? []
    let modelOverrideMapping = Dictionary(overrideModels.map { ($0.file, $0) }) { v, _ in v }
    let overrideLoras =
      (try? jsonDecoder.decode(
        [FailableDecodable<LoRAZoo.Specification>].self, from: override.loras
      ).compactMap({ $0.value })) ?? []
    let loraOverrideMapping = Dictionary(overrideLoras.map { ($0.file, $0) }) { v, _ in v }
    let hasImage = request.hasImage
    let shuffleCount = request.hints.reduce(0) {
      guard $1.hintType == "shuffle" else { return $0 }
      return $1.tensors.reduce(0) {
        $0 + ($1.weight > 0 ? 1 : 0)
      }
    }
    guard
      let cost = ComputeUnits.from(
        configuration, hasImage: hasImage, shuffleCount: shuffleCount,
        overrideMapping: (model: modelOverrideMapping, lora: loraOverrideMapping))
    else {
      logger.error(
        "Proxy Server can not calculate cost for configuration \(configuration)"
      )
      return (false, "Proxy Server can not calculate cost for model \(model)", model)
    }

    let (computeUnitPolicy, expirationTimestamp) =
      await controlConfigs.getComputeUnitPolicyAndExpirationTimestamp()
    let computeUnitPerBoost = await controlConfigs.computeUnitPerBoost
    let costThreshold = costThreshold(
      payload: payload,
      computeUnitPolicy: computeUnitPolicy,
      expirationTimestamp: expirationTimestamp,
      computeUnitPerBoost: computeUnitPerBoost
    )
    guard cost < costThreshold else {
      logger.error(
        "Proxy Server enqueue image generating request failed, cost \(cost) exceed threshold \(costThreshold)"
      )
      return (false, "cost \(cost) exceed threshold \(costThreshold)", model)
    }
    return (true, "", model)
  }

  func getEffectiveThrottlePolicy(
    for key: String, payload: JWTPayload, throttlePolicies: [String: Int]
  ) -> Int? {
    // Payg users get their own throttle limits
    if payload.consumableType == .payg || payload.consumableType == .paygFree {
      return throttlePolicies["\(key)_payg"] ?? throttlePolicies[key]
    }
    // Bridge users (SDK/API/HTTP) without boost get separate limits
    if payload.fromBridge == true && payload.consumableType != .boost {
      let bridgeKey =
        payload.userClass == .plus ? "\(key)_from_bridge_plus" : "\(key)_from_bridge"
      return throttlePolicies[bridgeKey] ?? throttlePolicies[key]
    }

    let priorityKey = payload.userClass == .plus ? "\(key)_plus" : key
    return throttlePolicies[priorityKey] ?? throttlePolicies[key]
  }

  func costThreshold(
    payload: JWTPayload, computeUnitPolicy: [String: Int]?, expirationTimestamp: Date?,
    computeUnitPerBoost: Int
  ) -> Int {
    let costThresholdFromPolicy = ComputeUnits.threshold(
      for: payload.userClass?.rawValue,
      computeUnitPolicy: computeUnitPolicy,
      expirationTimestamp: expirationTimestamp
    )
    let boostsToSpend = payload.amount ?? 0
    let costThresholdFromBoost = ComputeUnits.thresholdAfterBoost(
      policyThreshold: costThresholdFromPolicy, boostsToSpend: boostsToSpend)
    logger.info(
      "Proxy Server payload.amount: \(payload.amount as Any), computeUnitPerBoost: \(computeUnitPerBoost), generation id: \(payload.generationId as Any) and costThresholdFromBoost: \(costThresholdFromBoost)"
    )
    if costThresholdFromBoost > costThresholdFromPolicy {
      logger.info(
        "Proxy Server applying consumable threshold \(costThresholdFromBoost) for generation id: \(payload.generationId as Any)"
      )
    }
    return costThresholdFromBoost
  }

  func generateImage(
    request: ImageGenerationRequest,
    context: StreamingResponseCallContext<ImageGenerationResponse>
  ) -> EventLoopFuture<GRPCStatus> {
    let headers = context.headers
    guard let authorization = headers.first(name: "authorization"),
      let bearToken = parseBearer(from: authorization)
    else {
      return context.eventLoop.makeFailedFuture(
        GRPCStatus(code: .permissionDenied, message: "Service bear-token is empty")
      )
    }
    logger.info("generateImage request")

    var validationRequest = request
    validationRequest.contents = []
    let encodedBlob = try? validationRequest.serializedData()
    guard let encodedBlob = encodedBlob else {
      return context.eventLoop.makeFailedFuture(
        GRPCStatus(code: .permissionDenied, message: "Cannot encode validation request")
      )
    }
    let promise = context.eventLoop.makePromise(of: GRPCStatus.self)
    logger.info("Proxy Server enqueue image generating request")
    let logger = logger
    Task {
      let pem: String = await controlConfigs.publicKeyPEM
      let decoder = try? JWTDecoder(publicKeyPEM: pem)
      let payload = try? decoder?.decode(bearToken)
      guard let payload = payload else {
        logger.info("decode payload failed, bearToken:\(bearToken)")
        promise.fail(
          GRPCStatus(
            code: .permissionDenied, message: "Invalid payload"))
        return
      }
      let (isValidRequest, message, model) = await isValidRequest(
        payload: payload, encodedBlob: encodedBlob, request: request)
      guard isValidRequest, let model = model else {
        if payload.consumableType == .boost, let amount = payload.amount,
          let generationId = payload.generationId
        {
          logger.info(
            "isValidRequest cancel consumableType generationId:\( generationId), message:\(message)"
          )
          await proxyMessageSigner.completeBoost(
            action: .cancel, generationId: generationId, amount: amount,
            logger: logger)
        } else if payload.consumableType == .payg, let amount = payload.amount,
          let generationId = payload.generationId
        {
          logger.info(
            "isValidRequest cancel payg generationId:\( generationId), message:\(message)"
          )
          await proxyMessageSigner.completePayg(
            action: .cancel, generationId: generationId, amount: amount,
            logger: logger)
        } else {
          logger.info(
            "isValidRequest cancel generationId:\( payload.generationId as Any), without completeBoost, consumableType:\(payload.consumableType as Any), amount:\(payload.amount as Any), message:\(message)"
          )
        }
        promise.fail(
          GRPCStatus(
            code: .permissionDenied, message: message))
        return
      }
      let throttlePolicies = await controlConfigs.throttlePolicy
      let priority = taskPriority(
        from: payload.userClass, payload: payload,
        throttlePolicies: throttlePolicies)
      // Enqueue task.
      let responseWriter = WorkTaskResponseWriter(context: context)
      let heartbeat = Task {
        while !Task.isCancelled {
          // Abort the task if we cannot send response any more. Send empty response as heartbeat to keep Cloudflare alive.
          let _ = try await responseWriter.send(ImageGenerationResponse()).get()
          try? await Task.sleep(for: .seconds(10))  // Every 10 seconds send a heartbeat.
        }
      }
      let task = WorkTask(
        priority: priority, request: request, context: context, responseWriter: responseWriter, promise: promise,
        heartbeat: heartbeat, creationTimestamp: Date(), model: model, payload: payload)
      await taskQueue.addTask(task)
      if let worker = await taskQueue.nextWorker() {
        // Note that the extracted task may not be the ones we just enqueued.
        let highThreshold = throttlePolicies["high_free_worker_threshold"] ?? 8
        let communityThreshold = throttlePolicies["community_free_worker_threshold"] ?? 0
        let throttleQueueTimeoutSeconds = throttlePolicies["throttle_queue_timeout_seconds"] ?? 3600
        let taskLoopBreakoutSeconds = throttlePolicies["task_loop_breakout_seconds"] ?? 30
        if let nextTaskForWorker = await taskQueue.nextTaskForWorker(
          worker, highThreshold: highThreshold, communityThreshold: communityThreshold,
          taskLoopBreakoutSeconds: taskLoopBreakoutSeconds)
        {
          do {
            try await worker.executeTask(
              nextTaskForWorker, proxyMessageSigner: proxyMessageSigner,
              throttleQueueTimeoutSeconds: throttleQueueTimeoutSeconds)
            logger.info("Task execution completed successfully for worker \(worker.id)")
          } catch {
            logger.error("Task execution failed for worker \(worker.id): \(error)")
          }
        }
        await taskQueue.returnWorker(worker)
      } else {
        logger.error("worker stream finished, can not get available worker")
        heartbeat.cancel()
      }
    }

    return promise.futureResult
  }

  func taskPriority(
    from userClass: UserClass?, payload: JWTPayload, throttlePolicies: [String: Int]
  )
    -> ProxyTaskPriority
  {
    if payload.consumableType == .boost || payload.consumableType == .payg
      || payload.consumableType == .paygFree
    {
      return .real
    }

    // Check if user has exceeded 24-hour limit
    if let dayStat = payload.stats["24_hour"],
      let daySoftLimitLow = throttlePolicies["daily_soft_limit_low"],
      let daySoftLimitHigh = throttlePolicies["daily_soft_limit_high"],
      dayStat >= daySoftLimitLow
    {
      if userClass == .plus {
        if dayStat >= daySoftLimitHigh {
          logger.info("downgrade dt plus to background")
          return .background
        }
        logger.info("downgrade dt plus to low")
        return .low
      } else {
        return .background
      }
    }

    switch userClass {
    case .plus:
      return .high
    case .community:
      return .low
    case .background:
      return .background
    case .banned:
      return .background
    case .throttled:
      logger.error(" userid: \(payload.userId as Any), is throttled. priority as background")
      return .background
    default:
      return .low
    }
  }

  func filesExist(request: FileListRequest, context: StatusOnlyCallContext) -> EventLoopFuture<
    FileExistenceResponse
  > {
    let promise = context.eventLoop.makePromise(of: FileExistenceResponse.self)
    Task {
      let internalFilePath = await controlConfigs.modelListPath
      let response = FileExistenceResponse.with {
        $0.files = [String]()
        $0.existences = [Bool]()
        var fileList = [String]()
        if let fileContent = try? String(
          contentsOf: URL(fileURLWithPath: internalFilePath), encoding: .utf8)
        {
          fileList = fileContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
        } else {
          logger.error("Proxy Server file list is nil")
        }
        for file in request.files {
          $0.files.append(file)
          let existence = fileList.contains(file)
          $0.existences.append(existence)
        }
      }
      promise.succeed(response)
    }
    return promise.futureResult
  }

  public func pubkey(request: PubkeyRequest, context: StatusOnlyCallContext)
    -> EventLoopFuture<PubkeyResponse>
  {
    let promise = context.eventLoop.makePromise(of: PubkeyResponse.self)
    Task {
      let pubkey = await self.proxyMessageSigner.getPublicKey()
      let response = PubkeyResponse.with {
        if let pubkey = pubkey {
          $0.pubkey = pubkey
          $0.message = "get pubkey successfully"
        } else {
          $0.message = "failed to get pubkey"
        }
      }
      promise.succeed(response)
    }
    return promise.futureResult
  }

  public func hours(request: HoursRequest, context: any StatusOnlyCallContext) -> EventLoopFuture<
    HoursResponse
  > {
    let promise = context.eventLoop.makePromise(of: HoursResponse.self)
    Task {
      let (computeUnitPolicies, expirationTimestamp) =
        await controlConfigs.getComputeUnitPolicyAndExpirationTimestamp()

      let response = HoursResponse.with {

        if let expiration = expirationTimestamp, Date() < expiration {
          $0.thresholds = ComputeUnitThreshold.with {
            $0.community = Double(
              computeUnitPolicies["community"] ?? ComputeUnits.threshold(for: "community"))
            $0.plus = Double(computeUnitPolicies["plus"] ?? ComputeUnits.threshold(for: "plus"))
            $0.expireAt = Int64(expiration.timeIntervalSince1970)
          }
        }
      }
      promise.succeed(response)
    }
    return promise.futureResult
  }

  func echo(request: EchoRequest, context: StatusOnlyCallContext) -> EventLoopFuture<EchoReply> {
    let promise = context.eventLoop.makePromise(of: EchoReply.self)
    Task {
      let internalFilePath = await controlConfigs.modelListPath
      let (computeUnitPolicies, expirationTimestamp) =
        await controlConfigs.getComputeUnitPolicyAndExpirationTimestamp()

      let response = EchoReply.with {
        logger.info("Proxy Server Received echo from: \(request.name)")
        $0.message = "Hello, \(request.name)!"
        var fileList = [String]()
        if let fileContent = try? String(
          contentsOf: URL(fileURLWithPath: internalFilePath), encoding: .utf8)
        {
          fileList = fileContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
        } else {
          logger.error("Proxy Server file list is nil")
        }
        $0.files = fileList

        if let expiration = expirationTimestamp, Date() < expiration {
          $0.thresholds = ComputeUnitThreshold.with {
            $0.community = Double(
              computeUnitPolicies["community"] ?? ComputeUnits.threshold(for: "community"))
            $0.plus = Double(computeUnitPolicies["plus"] ?? ComputeUnits.threshold(for: "plus"))
            $0.expireAt = Int64(expiration.timeIntervalSince1970)
          }
        }
      }
      promise.succeed(response)
    }
    return promise.futureResult
  }

  func uploadFile(context: StreamingResponseCallContext<UploadResponse>) -> EventLoopFuture<
    (StreamEvent<FileUploadRequest>) -> Void
  > {
    context.statusPromise.fail(GRPCStatus(code: .unimplemented, message: "Service not supported"))
    return context.eventLoop.makeFailedFuture(
      GRPCStatus(code: .unimplemented, message: "Service not supported")
    )
  }
}

public class ProxyCPUServer {
  private let workers: [Worker]
  private let logger = Logger(label: "com.draw-things.image-generation-proxy-service")
  private var controlConfigs: ControlConfigs
  private var taskQueue: TaskQueue
  private var proxyMessageSigner: ProxyMessageSigner
  public init(
    workers: [Worker], publicKeyPEM: String, modelListPath: String, nonceSizeLimit: Int
  ) {
    self.workers = workers
    self.controlConfigs = ControlConfigs(
      throttlePolicy: [
        "15_min": 300, "10_min": 200, "5_min": 100, "1_hour": 1000, "1_min": 30,
        "24_hour_plus": 5000, "24_hour": 1000, "daily_soft_limit_low": 500,
        "daily_soft_limit_high": 750, "high_free_worker_threshold": 8,
        "community_free_worker_threshold": 0,
        "throttle_queue_timeout_seconds": 3600,
        "task_loop_breakout_seconds": 30,
        "worker_echo_rejoin_window_seconds": 1800,
        "worker_echo_rejoin_limit": 100,
        "24_hour_from_bridge_plus": 500, "24_hour_from_bridge": 100,
        "24_hour_payg": 50000,
        "15_min_payg": 3000, "10_min_payg": 2000, "5_min_payg": 1000,
        "1_hour_payg": 10000, "1_min_payg": 300,
      ], publicKeyPEM: publicKeyPEM, logger: logger, modelListPath: modelListPath,
      nonceSizeLimit: nonceSizeLimit)
    self.proxyMessageSigner = ProxyMessageSigner()
    self.taskQueue = TaskQueue(workers: workers, logger: logger)
  }

  public func startControlPanel(hosts: [String], port: Int) async throws {
    Task {
      logger.info("Control Panel Service starting on \(hosts) , port \(port)")
      let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
      let controlPanelService = ControlPanelService(
        taskQueue: taskQueue, controlConfigs: controlConfigs, logger: logger,
        proxyMessageSigner: proxyMessageSigner)

      var serverBindings: [EventLoopFuture<Server>] = []

      // Start all servers
      for host in hosts {
        let binding = Server.insecure(group: group)
          .withServiceProviders([controlPanelService])
          .bind(host: host, port: port)
        serverBindings.append(binding)
        logger.info("Control Panel Service started on \(host):\(port)")
      }

      // Wait for all servers to bind
      let servers = try await EventLoopFuture.whenAllSucceed(serverBindings, on: group.next()).get()

      // Wait for any server to close (first one that closes)
      let onCloseFutures = servers.map { $0.onClose }
      let _ = try await EventLoopFuture.whenAllComplete(onCloseFutures, on: group.next()).get()

      logger.info("Leave Control Panel Service, something may be wrong")
      try await group.syncShutdownGracefully()
    }
  }

  private static func certificatesFromPEMFile(_ path: String) throws -> [NIOSSLCertificate] {
    let pemContent = try String(contentsOfFile: path, encoding: .utf8)
    var certificates: [NIOSSLCertificate] = []

    // Split by certificate markers and clean up
    let pemComponents = pemContent.components(separatedBy: "-----BEGIN CERTIFICATE-----")
      .filter { !$0.isEmpty }
      .compactMap { component -> String? in
        guard let endMarkerRange = component.range(of: "-----END CERTIFICATE-----") else {
          return nil
        }
        let certContent = "-----BEGIN CERTIFICATE-----" + component[..<endMarkerRange.upperBound]
        // Ensure the certificate has both markers and proper content between them
        guard certContent.hasSuffix("-----END CERTIFICATE-----") else {
          return nil
        }
        return certContent.trimmingCharacters(in: .whitespacesAndNewlines)
      }

    for pemCert in pemComponents {
      let certData = Array(pemCert.utf8)
      guard let cert = try? NIOSSLCertificate(bytes: certData, format: .pem) else {
        continue
      }
      certificates.append(cert)
    }

    return certificates
  }

  public func startAndWait(
    host: String, port: Int, TLS: Bool, certPath: String, keyPath: String, numberOfThreads: Int,
    healthCheck: Bool
  )
    async throws
  {
    logger.info("ImageGenerationProxyService starting on \(host):\(port)")
    let group = MultiThreadedEventLoopGroup(numberOfThreads: numberOfThreads)
    let proxyService = ImageGenerationProxyService(
      taskQueue: taskQueue, controlConfigs: controlConfigs, logger: logger,
      healthCheck: healthCheck, proxyMessageSigner: proxyMessageSigner
    )
    let imageServer: Server
    if TLS {
      let certificates: [NIOSSLCertificate]
      let privateKey: NIOSSLPrivateKey
      if !certPath.isEmpty && !keyPath.isEmpty {
        certificates = try ProxyCPUServer.certificatesFromPEMFile(certPath)
        privateKey = try NIOSSLPrivateKey(file: keyPath, format: .pem)
      } else {
        certificates = [
          try NIOSSLCertificate(bytes: [UInt8](BinaryResources.server_crt_crt), format: .pem)
        ]
        privateKey = try NIOSSLPrivateKey(
          bytes: [UInt8](BinaryResources.server_key_key), format: .pem)
      }
      let certificateSources = certificates.map { NIOSSLCertificateSource.certificate($0) }
      imageServer = try Server.usingTLS(
        with: GRPCTLSConfiguration.makeServerConfigurationBackedByNIOSSL(
          certificateChain: certificateSources, privateKey: .privateKey(privateKey)),
        on: group
      )
      .withServiceProviders([proxyService])
      .withMaximumReceiveMessageLength(1024 * 1024 * 1024)
      .bind(host: host, port: port).wait()
    } else {

      imageServer = try Server.insecure(group: group)
        .withServiceProviders([proxyService])
        .withMaximumReceiveMessageLength(1024 * 1024 * 1024)
        .bind(host: host, port: port).wait()
    }

    logger.info("Image Generation Proxy Service started on port \(host):\(port)")
    try await imageServer.onClose.get()
    try await group.syncShutdownGracefully()
  }

}
