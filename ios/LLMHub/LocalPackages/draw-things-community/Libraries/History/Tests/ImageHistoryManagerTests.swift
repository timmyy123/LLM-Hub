import DataModels
import Foundation
import History
import SQLiteDflat
import XCTest

final class ImageHistoryManagerTests: XCTestCase {
  private var workspace: SQLiteWorkspace!
  private var temporaryDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
      "ImageHistoryManagerTests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: temporaryDirectory, withIntermediateDirectories: true, attributes: nil)
    workspace = SQLiteWorkspace(
      filePath: temporaryDirectory.appendingPathComponent("history.sqlite3").path,
      fileProtectionLevel: .noProtection)
  }

  override func tearDownWithError() throws {
    workspace.shutdown(flags: .truncate)
    workspace = nil
    try? FileManager.default.removeItem(at: temporaryDirectory)
    temporaryDirectory = nil
    try super.tearDownWithError()
  }

  private func onMain<T>(_ block: () -> T) -> T {
    if Thread.isMainThread {
      return block()
    }
    return DispatchQueue.main.sync(execute: block)
  }

  private func seedHistory(nodes: [TensorHistoryNode]) {
    let expectation = expectation(description: "seed-history")
    workspace.performChanges([TensorHistoryNode.self]) { transactionContext in
      for node in nodes {
        transactionContext.try(submit: TensorHistoryNodeChangeRequest.upsertRequest(node))
      }
    } completionHandler: { _ in
      DispatchQueue.main.async {
        expectation.fulfill()
      }
    }
    wait(for: [expectation], timeout: 5)
  }

  private func waitForCondition(timeout: TimeInterval = 2.0, _ condition: @escaping () -> Bool) {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
      if condition() { return }
      RunLoop.main.run(until: Date().addingTimeInterval(0.01))
    }
  }

  private func makeConfiguration(
    seed: UInt32 = 1, colorCalibration: DataModels.ColorCalibration = .disabled,
    expandPromptToJson: Bool = false
  ) -> GenerationConfiguration {
    GenerationConfiguration(
      id: 0, startWidth: 8, startHeight: 8, seed: seed, steps: 20, guidanceScale: 7.5,
      batchCount: 1, batchSize: 1, colorCalibration: colorCalibration,
      expandPromptToJson: expandPromptToJson
    )
  }

  private func makeHistory(
    configuration: GenerationConfiguration, tensorId: Int64?, isGenerated: Bool
  ) -> ImageHistoryManager.History {
    let imageData =
      tensorId.map {
        [
          ImageHistoryManager.ImageData(
            x: 0, y: 0, width: 512, height: 512, scaleFactorBy120: 120, tensorId: $0)
        ]
      } ?? []
    return ImageHistoryManager.History(
      imageData: imageData, reason: .generate, preview: nil, textEdits: nil, textLineage: nil,
      configuration: configuration, isGenerated: isGenerated, contentOffset: (x: 0, y: 0),
      scaleFactorBy120: 120, scriptSessionId: nil
    )
  }

  func testUnsacredSeekPersistsResolvedLineage() {
    seedHistory(
      nodes: [
        TensorHistoryNode(lineage: 2, logicalTime: 0, seed: 2),
        TensorHistoryNode(lineage: 3, logicalTime: 0, seed: 3),
      ])

    let manager = onMain {
      ImageHistoryManager(
        project: workspace, filePath: temporaryDirectory.appendingPathComponent("store.ckpt").path)
    }

    XCTAssertEqual(manager.lineage, 3)

    let isSacred = onMain {
      manager.seek(to: 0, lineage: 2)
    }

    XCTAssertFalse(isSacred)
    XCTAssertEqual(manager.lineage, 2)
    XCTAssertEqual(workspace.dictionary["image_seek_to_lineage", Int.self], 2)
  }

  func testSacredSeekClearsStoredLineage() {
    seedHistory(
      nodes: [
        TensorHistoryNode(lineage: 2, logicalTime: 0, seed: 2),
        TensorHistoryNode(lineage: 3, logicalTime: 0, seed: 3),
      ])

    let manager = onMain {
      ImageHistoryManager(
        project: workspace, filePath: temporaryDirectory.appendingPathComponent("store.ckpt").path)
    }

    _ = onMain {
      manager.seek(to: 0, lineage: 2)
    }
    XCTAssertEqual(workspace.dictionary["image_seek_to_lineage", Int.self], 2)

    _ = onMain {
      manager.seek(to: 0, lineage: nil)
    }

    XCTAssertNil(workspace.dictionary["image_seek_to_lineage", Int.self])
  }

  #if false
    // Disabled until ImageHistoryManager exposes generation-queue APIs again.
    func testPushGenerationQueueAddsPlaceholder() {
      let manager = onMain {
        ImageHistoryManager(
          project: workspace, filePath: temporaryDirectory.appendingPathComponent("store.ckpt").path
        )
      }

      let inProgressIdentifier = onMain {
        manager.pushGenerationQueue(
          makeHistory(configuration: makeConfiguration(), tensorId: nil, isGenerated: false)
        )
      }

      waitForCondition { [self] in
        self.onMain { Array(manager.allImageHistories(false)).count == 1 }
      }
      let nodes = onMain { Array(manager.allImageHistories(false)) }
      XCTAssertEqual(nodes.count, 1)
      XCTAssertEqual(nodes[0].wallClock, inProgressIdentifier)
      XCTAssertFalse(nodes[0].generated)
      XCTAssertEqual(nodes[0].dataStored, 0)
    }
  #endif

  func testColorCalibrationRoundTripsThroughHistory() {
    let manager = onMain {
      ImageHistoryManager(
        project: workspace, filePath: temporaryDirectory.appendingPathComponent("store.ckpt").path)
    }

    onMain {
      manager.pushHistory([
        makeHistory(
          configuration: makeConfiguration(colorCalibration: .lab), tensorId: 44,
          isGenerated: true)
      ])
    }

    waitForCondition { [self] in
      self.onMain { manager.configuration?.colorCalibration == .lab }
    }
    waitForCondition { [self] in
      self.onMain { manager.allImageHistories(false).first?.colorCalibration == .lab }
    }

    XCTAssertEqual(onMain { manager.configuration?.colorCalibration }, .lab)
    let nodes = onMain { Array(manager.allImageHistories(false)) }
    XCTAssertEqual(nodes.first?.colorCalibration.rawValue, DataModels.ColorCalibration.lab.rawValue)
  }

  func testSolAttentionRoundTripsThroughHistory() {
    let manager = onMain {
      ImageHistoryManager(
        project: workspace, filePath: temporaryDirectory.appendingPathComponent("store.ckpt").path)
    }
    var configuration = GenerationConfigurationBuilder(from: makeConfiguration())
    configuration.usesSolAttention = true
    configuration.solAttentionStart = 3
    configuration.solAttentionTau = 0.75
    onMain {
      manager.pushHistory([
        makeHistory(configuration: configuration.build(), tensorId: 46, isGenerated: true)
      ])
    }
    waitForCondition { [self] in
      self.onMain { manager.configuration?.usesSolAttention == true }
    }
    waitForCondition { [self] in
      self.onMain { manager.allImageHistories(false).first?.usesSolAttention == true }
    }
    XCTAssertEqual(onMain { manager.configuration?.usesSolAttention }, true)
    XCTAssertEqual(onMain { manager.configuration?.solAttentionStart }, 3)
    XCTAssertEqual(onMain { manager.configuration?.solAttentionTau }, 0.75)
    XCTAssertEqual(onMain { manager.allImageHistories(false).first?.usesSolAttention }, true)
    XCTAssertEqual(onMain { manager.allImageHistories(false).first?.solAttentionStart }, 3)
    XCTAssertEqual(onMain { manager.allImageHistories(false).first?.solAttentionTau }, 0.75)
  }

  func testExpandPromptToJsonRoundTripsThroughHistory() {
    let manager = onMain {
      ImageHistoryManager(
        project: workspace, filePath: temporaryDirectory.appendingPathComponent("store.ckpt").path)
    }

    onMain {
      manager.pushHistory([
        makeHistory(
          configuration: makeConfiguration(expandPromptToJson: true), tensorId: 45,
          isGenerated: true)
      ])
    }

    waitForCondition { [self] in
      self.onMain { manager.configuration?.expandPromptToJson == true }
    }
    waitForCondition { [self] in
      self.onMain { manager.allImageHistories(false).first?.expandPromptToJson == true }
    }

    XCTAssertEqual(onMain { manager.configuration?.expandPromptToJson }, true)
    XCTAssertEqual(onMain { manager.allImageHistories(false).first?.expandPromptToJson }, true)
  }

  #if false
    // Disabled until ImageHistoryManager exposes generation-queue APIs again.
    func testCompleteGenerationQueueFindsNodeAfterDeleteRenumber() {
      let manager = onMain {
        ImageHistoryManager(
          project: workspace, filePath: temporaryDirectory.appendingPathComponent("store.ckpt").path
        )
      }

      onMain {
        manager.pushHistory([
          makeHistory(configuration: makeConfiguration(seed: 11), tensorId: 11, isGenerated: true)
        ])
      }
      let inProgressIdentifier = onMain {
        manager.pushGenerationQueue(
          makeHistory(configuration: makeConfiguration(seed: 22), tensorId: nil, isGenerated: false)
        )
      }

      waitForCondition { [self] in
        self.onMain { manager.allImageHistories(false).contains(where: { $0.seed == 11 }) }
      }
      let firstNode = onMain { manager.allImageHistories(false).first(where: { $0.seed == 11 }) }
      guard let firstNode else {
        XCTFail("Expected first generated node")
        return
      }

      let deleteExpectation = expectation(description: "delete old history")
      onMain {
        manager.deleteHistory(firstNode) { _, _, _ in
          deleteExpectation.fulfill()
        }
      }
      wait(for: [deleteExpectation], timeout: 3.0)

      let finalized = onMain {
        manager.completeGenerationQueue(
          inProgressIdentifier,
          with: makeHistory(
            configuration: makeConfiguration(seed: 33), tensorId: 33, isGenerated: true)
        )
      }
      XCTAssertTrue(finalized)

      waitForCondition { [self] in
        self.onMain {
          guard
            let node = manager.allImageHistories(false).first(where: {
              $0.wallClock == inProgressIdentifier
            })
          else { return false }
          return node.generated && node.seed == 33 && node.dataStored == 1
        }
      }

      let nodes = onMain {
        Array(manager.allImageHistories(false))
      }
      guard let finalizedNode = nodes.first(where: { $0.wallClock == inProgressIdentifier }) else {
        XCTFail("Expected finalized queued node")
        return
      }
      XCTAssertTrue(finalizedNode.generated)
      XCTAssertEqual(finalizedNode.seed, 33)
      XCTAssertEqual(finalizedNode.dataStored, 1)
    }
  #endif
}
