import OrderedCollections

public struct GenerationProfile: Codable {
  public struct Timing: Codable {
    public var durations: [Double]
    public var name: String
    public init(durations: [Double], name: String) {
      self.durations = durations
      self.name = name
    }
  }
  public var timings: [Timing]
  public var duration: Double
  public init(timings: [Timing], duration: Double) {
    self.timings = timings
    self.duration = duration
  }
}

public struct GenerationProfileBuilder {
  private let totalStartTime: Double
  private var lastStartTime: Double
  private var skippedDuration: Double
  private var timings: OrderedDictionary<String, GenerationProfile.Timing>
  public init(_ currentTime: Double) {
    totalStartTime = currentTime
    lastStartTime = totalStartTime
    skippedDuration = 0
    timings = [:]
  }
  public mutating func append(_ name: String, _ currentTime: Double) {
    let duration = currentTime - lastStartTime
    var timing = timings[name, default: GenerationProfile.Timing(durations: [], name: name)]
    timing.durations.append(duration)
    timings[name] = timing
    lastStartTime = currentTime
  }
  public mutating func skip(duration: Double) {
    let duration = max(0, duration)
    skippedDuration += duration
    lastStartTime += duration
  }
  public func build(_ currentTime: Double) -> GenerationProfile {
    let duration = currentTime - totalStartTime - skippedDuration
    let timings = Array(timings.values)
    return GenerationProfile(timings: timings, duration: duration)
  }
}
