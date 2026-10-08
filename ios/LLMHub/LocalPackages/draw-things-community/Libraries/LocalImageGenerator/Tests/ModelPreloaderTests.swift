import DataModels
import Dflat
import Foundation
import SQLiteDflat
import WeightsCache
import XCTest

@testable import LocalImageGenerator

final class ModelPreloaderTests: XCTestCase {
  func testSettingsSubscriptionsReleasePreloaderAndWeightsCache() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let workspace = SQLiteWorkspace(
      filePath: directory.appendingPathComponent("config.sqlite3").path,
      fileProtectionLevel: .noProtection)
    defer { workspace.shutdown(flags: .truncate) }
    let configurations = workspace.fetch(for: GenerationConfiguration.self).where(
      GenerationConfiguration.id == 0, limit: .limit(0))
    let queue = DispatchQueue(label: "ModelPreloaderTests")
    weak var releasedPreloader: ModelPreloader?
    weak var releasedWeightsCache: WeightsCache?
    autoreleasepool {
      let weightsCache = WeightsCache(maxTotalCacheSize: 1_024, memorySubsystem: .UMA)
      let preloader = ModelPreloader(
        queue: queue, weightsCache: weightsCache, configurations: configurations,
        workspace: workspace)
      releasedPreloader = preloader
      releasedWeightsCache = weightsCache
      // Drain initial subscription delivery while the caller still owns the preloader.
      queue.sync {}
      withExtendedLifetime(preloader) {}
    }
    queue.sync {}
    // A live workspace must not keep the preloader or its weights cache alive.
    withExtendedLifetime(workspace) {
      XCTAssertNil(releasedPreloader)
      XCTAssertNil(releasedWeightsCache)
    }
  }
}
