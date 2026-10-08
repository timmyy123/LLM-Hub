import Dispatch
import ModelZoo

public struct OnlineModelLister {
  fileprivate static let queue = DispatchQueue(label: "online-model-lister-lock")
  fileprivate static var internalMemorizer = Set<String>()
  fileprivate static var models: [ModelZoo.Specification] = []
  fileprivate static var LoRAs: [LoRAZoo.Specification] = []
  fileprivate static var controlNets: [ControlNetZoo.Specification] = []
  fileprivate static var textualInversions: [TextualInversionZoo.Specification] = []
}

extension OnlineModelLister {
  public static var memorizer: Set<String> {
    return queue.sync {
      return internalMemorizer
    }
  }
  public static func setOnline(
    files: [String], models: [ModelZoo.Specification], LoRAs: [LoRAZoo.Specification],
    controlNets: [ControlNetZoo.Specification],
    textualInversions: [TextualInversionZoo.Specification]
  ) {
    dispatchPrecondition(condition: .onQueue(.main))
    queue.sync {
      Self.internalMemorizer = Set(files)
    }
    Self.models = models
    Self.LoRAs = LoRAs
    Self.controlNets = controlNets
    Self.textualInversions = textualInversions
    // Now set fallbacks.
    ModelZoo.fallbackMapping = Dictionary(ModelZoo.onlineSpecifications.map { ($0.file, $0) }) {
      v, _ in v
    }
    LoRAZoo.fallbackMapping = Dictionary(LoRAZoo.onlineSpecifications.map { ($0.file, $0) }) {
      v, _ in v
    }
    ControlNetZoo.fallbackMapping = Dictionary(
      ControlNetZoo.onlineSpecifications.map { ($0.file, $0) }
    ) { v, _ in v }
    TextualInversionZoo.fallbackMapping = Dictionary(
      TextualInversionZoo.onlineSpecifications.map { ($0.file, $0) }
    ) {
      v, _ in v
    }
  }
}

extension ModelZoo {
  public static var onlineSpecifications: [Specification] {
    dispatchPrecondition(condition: .onQueue(.main))
    let specifications = OnlineModelLister.models
    guard !specifications.isEmpty else { return [] }
    // Need to remove ones from official, and ones from community, and ones from local.
    let shown = Set(
      community.map({ $0.file })
        + availableSpecifications.compactMap({
          guard !isBuiltinModel($0.file) else {
            return $0.deprecated == true ? nil : $0.file
          }
          return isModelDownloaded($0) ? $0.file : nil
        }))
    return specifications.filter { !shown.contains($0.file) }
  }

  public static func isOnlineOnly(specification: Specification) -> Bool {
    dispatchPrecondition(condition: .onQueue(.main))
    return !isModelDownloaded(specification)
      && isModelDownloaded(specification, memorizedBy: OnlineModelLister.memorizer)
  }
}

extension LoRAZoo {
  public static var onlineSpecifications: [Specification] {
    dispatchPrecondition(condition: .onQueue(.main))
    let specifications = OnlineModelLister.LoRAs
    guard !specifications.isEmpty else { return [] }
    // Need to remove ones from official, and ones from community, and ones from local.
    let shown = Set(
      community.map({ $0.file })
        + availableSpecifications.compactMap({
          guard !isBuiltinLoRA($0.file) else {
            return $0.deprecated == true ? nil : $0.file
          }
          return isModelDownloaded($0) ? $0.file : nil
        }))
    return specifications.filter { !shown.contains($0.file) }
  }

  public static func isOnlineOnly(specification: Specification) -> Bool {
    dispatchPrecondition(condition: .onQueue(.main))
    return !isModelDownloaded(specification)
      && isModelDownloaded(specification, memorizedBy: OnlineModelLister.memorizer)
  }
}

extension ControlNetZoo {
  public static var onlineSpecifications: [Specification] {
    dispatchPrecondition(condition: .onQueue(.main))
    let specifications = OnlineModelLister.controlNets
    guard !specifications.isEmpty else { return [] }
    // Need to remove ones from official, and ones from community, and ones from local.
    let shown = Set(
      community.map({ $0.file })
        + availableSpecifications.compactMap({
          guard !isBuiltinControl($0.file) else {
            return $0.deprecated == true ? nil : $0.file
          }
          return isModelDownloaded($0) ? $0.file : nil
        }))
    return specifications.filter { !shown.contains($0.file) }
  }

  public static func isOnlineOnly(specification: Specification) -> Bool {
    dispatchPrecondition(condition: .onQueue(.main))
    return !isModelDownloaded(specification)
      && isModelDownloaded(specification, memorizedBy: OnlineModelLister.memorizer)
  }
}

extension TextualInversionZoo {
  public static var onlineSpecifications: [Specification] {
    dispatchPrecondition(condition: .onQueue(.main))
    let specifications = OnlineModelLister.textualInversions
    guard !specifications.isEmpty else { return [] }
    // Need to remove ones from official, and ones from community, and ones from local.
    let shown: Set<String> = Set(
      community.map({ $0.file })
        + availableSpecifications.compactMap({
          guard !isBuiltinTextualInversion($0.file) else {
            return $0.deprecated == true ? nil : $0.file
          }
          return isModelDownloaded($0.file) ? $0.file : nil
        }))
    return specifications.filter { !shown.contains($0.file) }
  }

  public static func isOnlineOnly(specification: Specification) -> Bool {
    dispatchPrecondition(condition: .onQueue(.main))
    return !isModelDownloaded(specification.file)
      && isModelDownloaded(specification.file, memorizedBy: OnlineModelLister.memorizer)
  }
}
