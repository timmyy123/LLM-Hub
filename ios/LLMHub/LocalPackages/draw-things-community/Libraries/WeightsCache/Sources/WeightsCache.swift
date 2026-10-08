import Collections
import NNC

public final class WeightsCache {
  public enum MemorySubsystem {
    case dGPU
    case UMA
  }
  struct Item: Comparable {
    var file: String
    var weights: [(key: String, value: AnyTensor)]
    var size: UInt64
    var timestamp: Int64
  }
  private var heap: Heap<Item>
  private var map: [String: Item]
  public var maxTotalCacheSize: UInt64 {  // Maximum number of items in the cache
    didSet {
      guard maxTotalCacheSize < oldValue else { return }
      evict(for: 0)  // Trigger eviction logic.
    }
  }
  public let memorySubsystem: MemorySubsystem
  private var timestamp: Int64
  private var currentTotalSize: UInt64  // Optional: if you also have a total size limit

  public init(maxTotalCacheSize: UInt64, memorySubsystem: MemorySubsystem) {
    self.maxTotalCacheSize = maxTotalCacheSize
    self.memorySubsystem = memorySubsystem
    heap = Heap<Item>()
    map = [:]
    currentTotalSize = 0  // Initialize if tracking total size
    timestamp = 0
  }

  public subscript(file: String) -> (size: UInt64, weights: [(key: String, value: AnyTensor)])? {
    get {
      guard let item = map[file] else { return nil }
      return (size: item.size, weights: item.weights)
    }
    set {
      guard let newValue = newValue else {
        if let index = map.index(forKey: file) {
          let item = map.remove(at: index).value
          // If this exists, now loop through heap to remove.
          heap = Heap(map.values)
          currentTotalSize -= item.size
        }
        return
      }
      // If this item itself is larger than total cache size, we cannot fit.
      if newValue.size > maxTotalCacheSize {
        return
      }
      // If this item size is smaller than smallest in the cache, and we are at capacity, there is no need to insert.
      if let min = heap.min, newValue.size < min.size,
        newValue.size + currentTotalSize > maxTotalCacheSize
      {
        return
      }
      // If this item size plus the current size is larger, need to evict first.
      evict(for: newValue.size)
      timestamp += 1
      let item = Item(
        file: file, weights: newValue.weights, size: newValue.size, timestamp: timestamp)
      map[file] = item
      heap.insert(item)
      currentTotalSize += newValue.size
    }
  }

  public func remove(at file: String) -> (size: UInt64, weights: [(key: String, value: AnyTensor)])?
  {
    guard let index = map.index(forKey: file) else { return nil }
    let item = map.remove(at: index).value
    heap = Heap(map.values)
    currentTotalSize -= item.size
    return (size: item.size, weights: item.weights)
  }

  private func updateTimestamp(for file: String) {
    timestamp += 1
    map[file]?.timestamp = timestamp
    heap = Heap(map.values)
  }

  public func removeAll() {
    map.removeAll()
    heap = Heap()
  }

  // Evict the smallest item if the cache is over capacity
  private func evict(for size: UInt64) {
    // Define `maxTotalCacheSize` for this logic
    while currentTotalSize + size > maxTotalCacheSize {
      if let smallestItem = heap.popMin() {
        map[smallestItem.file] = nil
        currentTotalSize -= smallestItem.size
      } else {
        break  // Heap is empty
      }
    }
  }

  // Get the current number of items in the cache
  public var count: Int {
    return heap.count
  }

  // Get the current total size of items in the cache
  public var totalSize: UInt64 {
    return currentTotalSize
  }
}

extension WeightsCache.Item {
  // Conformance to Comparable for Min-Heap behavior based on 'size'
  static func < (lhs: WeightsCache.Item, rhs: WeightsCache.Item) -> Bool {
    // To make it a min-heap (smallest item at the top),
    // an item is "less than" another if its size is smaller.
    return lhs.size < rhs.size || (lhs.size == rhs.size && lhs.timestamp < rhs.timestamp)
  }

  // Conformance to Equatable (required by Comparable)
  static func == (lhs: WeightsCache.Item, rhs: WeightsCache.Item) -> Bool {
    return lhs.file == rhs.file && lhs.size == rhs.size
  }
}

extension WeightsCache {
  public func detach(_ file: String, to parameters: @autoclosure () -> Model.Parameters) -> Bool {
    switch memorySubsystem {
    case .UMA:
      guard var weights = remove(at: file)?.weights else { return false }
      let parameters = parameters()
      parameters.attach(consuming: &weights)
    case .dGPU:
      guard let weights = self[file]?.weights else { return false }
      let parameters = parameters()
      parameters.attach(from: weights)
    }
    return true
  }

  public func attach(_ file: String, from parameters: @autoclosure () -> Model.Parameters) {
    guard maxTotalCacheSize > 0 else { return }
    switch memorySubsystem {
    case .UMA:
      let parameters = parameters()
      var size = parameters.size  // Make sure we grab size prior to detach.
      let weights = parameters.detach(.GPU(0)).filter {
        guard !$0.key.contains("lora_down") && !$0.key.contains("lora_up") else {
          size = UInt64(max(Int(size) - $0.value.size, 0))
          return false
        }
        return true
      }
      self[file] = (size: size, weights: weights)
    case .dGPU:
      guard self[file] == nil else {
        // If already exists, nothing to attach. But we need to update the timestamp.
        updateTimestamp(for: file)
        return
      }
      let parameters = parameters()
      var size = parameters.size  // Make sure we grab size prior to detach.
      // Otherwise copy to CPU.
      let weights = parameters.detach(.CPU).filter {
        guard !$0.key.contains("lora_down") && !$0.key.contains("lora_up") else {
          size = UInt64(max(Int(size) - $0.value.size, 0))
          return false
        }
        return true
      }
      self[file] = (size: size, weights: weights)
    }
  }
}
