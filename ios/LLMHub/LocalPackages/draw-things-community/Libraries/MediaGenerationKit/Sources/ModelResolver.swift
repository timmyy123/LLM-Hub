import Foundation
import GRPCServer
import ModelZoo

/// Resolved model metadata returned by ``MediaGenerationEnvironment`` helpers.
public struct MediaGenerationResolvedModel: Sendable, Hashable {
  public var file: String
  public var name: String
  public var description: String
  public var version: String?
  public var huggingFaceLink: String?
  public var isDownloaded: Bool

  public init(
    file: String,
    name: String,
    description: String,
    version: String? = nil,
    huggingFaceLink: String? = nil,
    isDownloaded: Bool = false
  ) {
    self.file = file
    self.name = name
    self.description = description
    self.version = version
    self.huggingFaceLink = huggingFaceLink
    self.isDownloaded = isDownloaded
  }
}

internal enum ModelZooSource {
  case builtin
  case bundled
  case remote
}

internal enum ModelZooLoader {
  private static let remoteURL = URL(string: "https://models.drawthings.ai/models.json")!
  private struct State {
    var bundledCache: [ModelZoo.Specification]?
    var remoteCache: [ModelZoo.Specification]?
    var remoteLoadTask: Task<[ModelZoo.Specification], Never>?
  }

  private static var state = ProtectedValue(State())

  static func specificationsSync(from source: ModelZooSource, offline: Bool) -> [ModelZoo.Specification]
  {
    switch source {
    case .builtin:
      return ModelZoo.availableSpecifications.filter { $0.remoteApiModelConfig == nil }
    case .bundled:
      var bundledCache: [ModelZoo.Specification]?
      state.modify { state in
        bundledCache = state.bundledCache
      }
      if let bundledCache { return bundledCache }
      let loaded = MediaGenerationResourceLoader.bundledData(resource: "models").flatMap(parse) ?? []
      var resolved = loaded
      state.modify { state in
        if let cached = state.bundledCache {
          resolved = cached
        } else {
          state.bundledCache = loaded
        }
      }
      return resolved
    case .remote:
      guard !offline else { return [] }
      var remoteCache: [ModelZoo.Specification]?
      state.modify { state in
        remoteCache = state.remoteCache
      }
      if let remoteCache { return remoteCache }
      let loaded = MediaGenerationResourceLoader.fetchRemoteData(url: remoteURL).flatMap(parse) ?? []
      var resolved = loaded
      state.modify { state in
        if let cached = state.remoteCache {
          resolved = cached
        } else {
          state.remoteCache = loaded
        }
      }
      return resolved
    }
  }

  static func specifications(from source: ModelZooSource, offline: Bool) async -> [ModelZoo.Specification]
  {
    switch source {
    case .builtin:
      return specificationsSync(from: .builtin, offline: offline)
    case .bundled:
      return specificationsSync(from: .bundled, offline: offline)
    case .remote:
      guard !offline else { return [] }
      let remoteState = cachedRemoteState()
      if let remoteCache = remoteState.cache {
        return remoteCache
      }
      if let remoteLoadTask = remoteState.task {
        return await remoteLoadTask.value
      }
      let loadTask = Task<[ModelZoo.Specification], Never> {
        guard let data = await MediaGenerationResourceLoader.fetchRemoteData(url: remoteURL) else {
          return []
        }
        return parse(data)
      }
      updateRemoteState(cache: nil, task: loadTask)

      let loaded = await loadTask.value

      updateRemoteState(cache: loaded, task: nil)
      return loaded
    }
  }

  private static func cachedRemoteState() -> (
    cache: [ModelZoo.Specification]?, task: Task<[ModelZoo.Specification], Never>?
  ) {
    var cache: [ModelZoo.Specification]?
    var task: Task<[ModelZoo.Specification], Never>?
    state.modify { state in
      cache = state.remoteCache
      task = state.remoteLoadTask
    }
    return (cache, task)
  }

  static func cachedRemoteSpecifications() -> [ModelZoo.Specification]? {
    var remoteCache: [ModelZoo.Specification]?
    state.modify { state in
      remoteCache = state.remoteCache
    }
    return remoteCache
  }

  private static func updateRemoteState(
    cache: [ModelZoo.Specification]?,
    task: Task<[ModelZoo.Specification], Never>?
  ) {
    state.modify { state in
      if let cache {
        state.remoteCache = cache
      }
      state.remoteLoadTask = task
    }
  }

  private static func parse(_ data: Data) -> [ModelZoo.Specification] {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let specifications = try? decoder.decode(
        [FailableDecodable<ModelZoo.Specification>].self,
        from: data
      ).compactMap(\.value)
    else {
      return []
    }
    return specifications.filter { $0.remoteApiModelConfig == nil }
  }
}

internal enum ModelResolver {
  typealias Model = MediaGenerationResolvedModel

  static func resolve(_ input: String, offline: Bool, operation: String = "resolveModel") throws
    -> Model?
  {
    if let specification = try specification(for: input, offline: offline, operation: operation) {
      return model(from: specification)
    }
    return nil
  }

  static func resolve(_ input: String, offline: Bool) async throws -> Model? {
    if let specification = await specification(for: input, offline: offline) {
      return model(from: specification)
    }
    return nil
  }

  static func specification(for input: String, offline: Bool, operation: String = "resolveModel")
    throws -> ModelZoo.Specification?
  {
    if let specification = ModelZoo.resolveModelReference(input)?.specification {
      return specification
    }

    if let bundled = matchingSpecification(
      for: input,
      in: ModelZooLoader.specificationsSync(from: .bundled, offline: offline)
    ) {
      primeOverrideMapping(with: bundled)
      return bundled
    }

    if offline {
      return nil
    }

    if let remoteSpecifications = ModelZooLoader.cachedRemoteSpecifications() {
      if let remote = matchingSpecification(
        for: input,
        in: remoteSpecifications
      ) {
        primeOverrideMapping(with: remote)
        return remote
      }
      return nil
    }

    throw MediaGenerationKitError.asyncOperationRequired(operation)
  }

  static func specification(for input: String, offline: Bool) async -> ModelZoo.Specification? {
    if let specification = ModelZoo.resolveModelReference(input)?.specification {
      return specification
    }

    if let bundled = matchingSpecification(
      for: input,
      in: ModelZooLoader.specificationsSync(from: .bundled, offline: offline)
    ) {
      primeOverrideMapping(with: bundled)
      return bundled
    }

    if let remote = matchingSpecification(
      for: input,
      in: await ModelZooLoader.specifications(from: .remote, offline: offline)
    ) {
      primeOverrideMapping(with: remote)
      return remote
    }

    return nil
  }

  static func suggestions(_ input: String, limit: Int = 5, offline: Bool) throws -> [Model] {
    if !offline, ModelZooLoader.cachedRemoteSpecifications() == nil {
      throw MediaGenerationKitError.asyncOperationRequired("suggestedModels")
    }

    var results = [Model]()
    var seen = Set<String>()

    for specification in ModelZoo.candidateSpecifications(forModelReference: input, limit: limit) {
      let candidate = model(from: specification)
      if seen.insert(candidate.file).inserted {
        results.append(candidate)
      }
    }

    for source in [ModelZooSource.bundled, .remote] {
      guard results.count < limit else { break }
      let specifications: [ModelZoo.Specification]
      switch source {
      case .bundled:
        specifications = ModelZooLoader.specificationsSync(from: .bundled, offline: offline)
      case .remote:
        specifications = offline ? [] : (ModelZooLoader.cachedRemoteSpecifications() ?? [])
      case .builtin:
        specifications = ModelZooLoader.specificationsSync(from: .builtin, offline: offline)
      }
      let candidates = candidateSpecifications(
        for: input,
        in: specifications,
        limit: limit - results.count
      )
      for specification in candidates {
        let candidate = model(from: specification)
        if seen.insert(candidate.file).inserted {
          results.append(candidate)
        }
      }
    }

    return Array(results.prefix(limit))
  }

  static func suggestions(_ input: String, limit: Int = 5, offline: Bool) async -> [Model] {
    var results = [Model]()
    var seen = Set<String>()

    for specification in ModelZoo.candidateSpecifications(forModelReference: input, limit: limit) {
      let candidate = model(from: specification)
      if seen.insert(candidate.file).inserted {
        results.append(candidate)
      }
    }

    guard results.count < limit else {
      return Array(results.prefix(limit))
    }

    let bundledCandidates = candidateSpecifications(
      for: input,
      in: ModelZooLoader.specificationsSync(from: .bundled, offline: offline),
      limit: limit - results.count
    )
    for specification in bundledCandidates {
      let candidate = model(from: specification)
      if seen.insert(candidate.file).inserted {
        results.append(candidate)
      }
    }

    guard results.count < limit else {
      return Array(results.prefix(limit))
    }

    let remoteCandidates = candidateSpecifications(
      for: input,
      in: await ModelZooLoader.specifications(from: .remote, offline: offline),
      limit: limit - results.count
    )
    for specification in remoteCandidates {
      let candidate = model(from: specification)
      if seen.insert(candidate.file).inserted {
        results.append(candidate)
      }
    }

    return Array(results.prefix(limit))
  }

  static func model(from specification: ModelZoo.Specification) -> Model {
    Model(
      file: specification.file,
      name: specification.name,
      description: specification.note ?? "",
      version: ModelZoo.humanReadableNameForVersion(specification.version),
      huggingFaceLink: specification.huggingFaceLink,
      isDownloaded: ModelZoo.isModelDownloaded(specification.file)
    )
  }

  static func catalogModels(includeDownloaded: Bool, offline: Bool) -> [MediaGenerationResolvedModel] {
    var deduplicated = [String: ModelZoo.Specification]()
    for source in [ModelZooSource.builtin, .bundled] {
      for specification in ModelZooLoader.specificationsSync(from: source, offline: offline) {
        deduplicated[specification.file] = specification
      }
    }

    if offline {
      return deduplicated.values
        .map(model(from:))
        .filter { includeDownloaded || !$0.isDownloaded }
        .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    if let remoteSpecifications = ModelZooLoader.cachedRemoteSpecifications() {
      for specification in remoteSpecifications {
        deduplicated[specification.file] = specification
      }
    } else {
      // Callers that need network-backed remote catalog data must use the async overload.
      // Sync callers are expected to throw before reaching here.
    }

    return deduplicated.values
      .map(model(from:))
      .filter { includeDownloaded || !$0.isDownloaded }
      .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
  }

  static func catalogModels(includeDownloaded: Bool, offline: Bool) async
    -> [MediaGenerationResolvedModel]
  {
    var deduplicated = [String: ModelZoo.Specification]()
    for source in [ModelZooSource.builtin, .bundled, .remote] {
      for specification in await ModelZooLoader.specifications(from: source, offline: offline) {
        deduplicated[specification.file] = specification
      }
    }

    return deduplicated.values
      .map(model(from:))
      .filter { includeDownloaded || !$0.isDownloaded }
      .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
  }

  static func catalogModelsSynchronouslyIfAvailable(includeDownloaded: Bool, offline: Bool) throws
    -> [MediaGenerationResolvedModel]
  {
    if !offline, ModelZooLoader.cachedRemoteSpecifications() == nil {
      throw MediaGenerationKitError.asyncOperationRequired("downloadableModels")
    }
    return catalogModels(includeDownloaded: includeDownloaded, offline: offline)
  }

  private static func matchingSpecification(
    for input: String,
    in specifications: [ModelZoo.Specification]
  ) -> ModelZoo.Specification? {
    if let exactFileMatch = specifications.first(where: { $0.file == input }) {
      return exactFileMatch
    }
    if let exactNameMatch = specifications.first(where: { $0.name == input }) {
      return exactNameMatch
    }
    guard let canonicalRepo = ModelZoo.normalizeHuggingFaceRepo(input) else {
      return nil
    }
    return specifications.first { specification in
      guard let huggingFaceLink = specification.huggingFaceLink,
        let specificationRepo = ModelZoo.normalizeHuggingFaceRepo(huggingFaceLink)
      else {
        return false
      }
      return specificationRepo == canonicalRepo
    }
  }

  private static func candidateSpecifications(
    for input: String,
    in specifications: [ModelZoo.Specification],
    limit: Int
  ) -> [ModelZoo.Specification] {
    let query = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !query.isEmpty else { return [] }
    return specifications
      .filter {
        $0.file.lowercased().contains(query)
          || $0.name.lowercased().contains(query)
          || ($0.huggingFaceLink?.lowercased().contains(query) ?? false)
      }
      .sorted {
        let lhsExact = $0.file.caseInsensitiveCompare(input) == .orderedSame
          || $0.name.caseInsensitiveCompare(input) == .orderedSame
        let rhsExact = $1.file.caseInsensitiveCompare(input) == .orderedSame
          || $1.name.caseInsensitiveCompare(input) == .orderedSame
        if lhsExact != rhsExact {
          return lhsExact
        }
        return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
      }
      .prefix(max(0, limit))
      .map { $0 }
  }

  private static func primeOverrideMapping(with specification: ModelZoo.Specification) {
    ModelZoo.overrideMapping[specification.file] = specification
    if let huggingFaceLink = specification.huggingFaceLink,
      let repo = ModelZoo.normalizeHuggingFaceRepo(huggingFaceLink)
    {
      ModelZoo.huggingFaceRepoOverrideMapping[repo] = specification
    }
  }
}
