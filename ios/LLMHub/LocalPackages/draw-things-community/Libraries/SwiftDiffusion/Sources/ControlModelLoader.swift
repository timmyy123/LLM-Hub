import NNC

public struct ControlModelLoader<FloatType: BinaryFloatingPoint & TensorNumeric> {
  private var stores: [(DynamicGraph.Store, keys: Set<String>)]
  private init(stores: [(DynamicGraph.Store, keys: Set<String>)]) {
    self.stores = stores
  }
  public func loadMergedWeight(name: String) -> DynamicGraph.Store.ModelReaderResult? {
    guard stores.count > 0 else { return nil }
    if let store = stores.first(where: { $0.keys.contains(name) }) {
      return .continue(name, store: store.0)  // So it will continue load from this store.
    }
    guard let last = name.components(separatedBy: "[").last, last.count > 2 else { return nil }
    // Remove t-.
    let name = last.suffix(from: last.index(last.startIndex, offsetBy: 2))
    let parts = name.components(separatedBy: ".")
    guard parts.count >= 2, let index = Int(parts[0]), index >= 0 && index < stores.count else {
      return nil
    }
    guard
      let tensor = stores[index].0.read(
        "__pulid__[t-" + parts[1], codec: [.ezm7, .externalData, .q6p, .q8p])
    else { return nil }
    if parts[1].contains("norm2") {
      return .final(Tensor<Float>(from: tensor))
    } else {
      return .final(tensor)
    }
  }
}

extension ControlModelLoader {
  private static func _openStore(
    _ graph: DynamicGraph, index: Int, injectControlModels: [ControlModel<FloatType>],
    stores: [(DynamicGraph.Store, keys: Set<String>)], version: ModelVersion,
    handler: (ControlModelLoader) -> Void
  ) {
    guard index < injectControlModels.count else {
      handler(ControlModelLoader(stores: stores))
      return
    }
    for i in index..<injectControlModels.count {
      // Only open for PuLID of FLUX.1, or VACE.
      let isPuLID = injectControlModels[i].type == .pulid && version == .flux1
      let isVACE =
        injectControlModels[i].type == .controlnet
        && (version == .wan21_14b || version == .wan21_1_3b)
      if isPuLID || isVACE {
        graph.openStore(
          injectControlModels[i].filePaths[0], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: injectControlModels[i].filePaths[0])
        ) { store in
          let keys: Set<String>
          if isVACE {
            keys = Set(store.keys)
          } else {
            keys = Set<String>()
          }
          _openStore(
            graph, index: i + 1, injectControlModels: injectControlModels,
            stores: stores + [(store, keys: keys)],
            version: version, handler: handler)
        }
        return
      }
    }
    handler(ControlModelLoader(stores: stores))
  }
  public static func openStore(
    _ graph: DynamicGraph, injectControlModels: [ControlModel<FloatType>], version: ModelVersion,
    handler: (ControlModelLoader) -> Void
  ) {
    _openStore(
      graph, index: 0, injectControlModels: injectControlModels, stores: [], version: version,
      handler: handler)
  }
}
