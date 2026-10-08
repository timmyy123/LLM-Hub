import Atomics
import Diffusion
import Foundation
import GRPCServer

internal struct MediaGenerationExecutionHintImage {
  let data: Data
  let weight: Float
}

internal struct MediaGenerationExecutionHint {
  let type: ControlHintType
  let images: [MediaGenerationExecutionHintImage]
}

internal struct MediaGenerationExecutionInputs {
  let image: Data?
  let mask: Data?
  let hints: [MediaGenerationExecutionHint]
}

internal final class MediaGenerationCancellationBridge {
  private struct State {
    var cancellation: (() -> Void)?
    var completed = false
  }

  private let cancelled = ManagedAtomic(false)
  private var state = ProtectedValue(State())

  var isCancelled: Bool {
    cancelled.load(ordering: .relaxed)
  }

  func setCancellation(_ cancellation: @escaping () -> Void) {
    var isCompleted = false
    state.modify { state in
      if state.completed {
        isCompleted = true
        return
      }
      state.cancellation = cancellation
    }
    if isCompleted {
      return
    }

    if cancelled.load(ordering: .acquiring) {
      cancellation()
    }
  }

  func cancel() {
    var shouldCancelImmediately = false
    var cancellation: (() -> Void)?
    state.modify { state in
      if state.completed {
        return
      }
      cancellation = state.cancellation
    }

    cancelled.store(true, ordering: .releasing)
    shouldCancelImmediately = cancellation != nil
    if shouldCancelImmediately {
      cancellation?()
    }
  }

  func finish() {
    state.modify { state in
      state.completed = true
      state.cancellation = nil
    }
  }
}

internal final class MediaGenerationAsyncResultBridge<Value>: @unchecked Sendable {
  private struct State {
    var continuation: CheckedContinuation<Value, Error>?
    var finished = false
  }

  private var state = ProtectedValue(State())

  @discardableResult
  func install(_ continuation: CheckedContinuation<Value, Error>) -> Bool {
    var alreadyFinished = false
    state.modify { state in
      if state.finished {
        alreadyFinished = true
        return
      }
      state.continuation = continuation
    }
    if alreadyFinished {
      continuation.resume(throwing: CancellationError())
      return false
    }
    return true
  }

  func resume(returning value: Value) {
    resume(with: .success(value))
  }

  func resume(throwing error: Error) {
    resume(with: .failure(error))
  }

  func resume(with result: Result<Value, Error>) {
    var continuation: CheckedContinuation<Value, Error>?
    var shouldResume = false
    state.modify { state in
      if state.finished {
        return
      }
      state.finished = true
      continuation = state.continuation
      state.continuation = nil
      shouldResume = true
    }
    if shouldResume {
      continuation?.resume(with: result)
    }
  }

  func cancel() {
    resume(throwing: CancellationError())
  }
}
