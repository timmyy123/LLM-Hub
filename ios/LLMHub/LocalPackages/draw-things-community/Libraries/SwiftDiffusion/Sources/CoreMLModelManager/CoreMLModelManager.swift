import Atomics
import Foundation

#if os(Linux)
  typealias os_unfair_lock_s = pthread_mutex_t

  func os_unfair_lock() -> os_unfair_lock_s {
    var lock = os_unfair_lock_s()
    pthread_mutex_init(&lock, nil)
    return lock
  }

  func os_unfair_lock_lock(_ lock: UnsafeMutablePointer<os_unfair_lock_s>) {
    pthread_mutex_lock(lock)
  }

  func os_unfair_lock_unlock(_ lock: UnsafeMutablePointer<os_unfair_lock_s>) {
    pthread_mutex_unlock(lock)
  }

#endif

public struct CoreMLModelManager {
  public static let maxNumberOfConvertedModels = ManagedAtomic(3)
  public static let reduceMemoryFor2x = ManagedAtomic(false)
  public static let reduceMemoryFor1x = ManagedAtomic(false)
  public static let isCoreMLSupported = ManagedAtomic(false)
  public static let isLoRASupported = ManagedAtomic(false)
  public static let computeUnits = ManagedAtomic(0)  // 0 - cpuAndNeuralEngine, 1 - cpuAndGPU, 2 - all
  static var lock = os_unfair_lock()
  static var modelConverted = Set<String>()
  public static func isModelConverted(_ model: String) -> Bool {
    os_unfair_lock_lock(&lock)
    let isModelConverted = modelConverted.contains(model)
    os_unfair_lock_unlock(&lock)
    return isModelConverted
  }
  public static func setModelConverted(_ model: String) {
    os_unfair_lock_lock(&lock)
    modelConverted.insert(model)
    os_unfair_lock_unlock(&lock)
  }
  public static func removeModelsConverted(_ models: [String]) {
    os_unfair_lock_lock(&lock)
    modelConverted.subtract(models)
    os_unfair_lock_unlock(&lock)
  }
  public static func removeAllConvertedModels() {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let coreMLUrl = urls.first!.appendingPathComponent("coreml")
    try? fileManager.removeItem(at: coreMLUrl)
    os_unfair_lock_lock(&lock)
    modelConverted.removeAll()
    os_unfair_lock_unlock(&lock)
  }
}
