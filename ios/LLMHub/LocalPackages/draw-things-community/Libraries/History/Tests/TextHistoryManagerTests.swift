import Foundation
import SQLiteDflat
import TextHistory
import XCTest

final class TextHistoryManagerTests: XCTestCase {
  private var workspace: SQLiteWorkspace!
  private var temporaryDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
      "TextHistoryManagerTests-\(UUID().uuidString)")
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

  private func seedHistory(
    textHistories: [TextHistoryNode], lineages: [TextLineageNode] = []
  ) {
    let expectation = expectation(description: "seed-history")
    workspace.performChanges([TextHistoryNode.self, TextLineageNode.self]) { transactionContext in
      for textHistory in textHistories {
        transactionContext.try(submit: TextHistoryNodeChangeRequest.upsertRequest(textHistory))
      }
      for lineage in lineages {
        transactionContext.try(submit: TextLineageNodeChangeRequest.upsertRequest(lineage))
      }
    } completionHandler: { _ in
      DispatchQueue.main.async {
        expectation.fulfill()
      }
    }
    wait(for: [expectation], timeout: 5)
  }

  func testInitialStateAndSeekToEmptyPrompt() {
    let manager = onMain {
      TextHistoryManager(project: workspace, initialPositiveText: "hello")
    }

    XCTAssertEqual(manager.currentPositiveText, "hello")
    XCTAssertEqual(manager.currentNegativeText, "")
    XCTAssertEqual(manager.currentEdits, 1)
    XCTAssertEqual(manager.maxEdits, 1)

    onMain {
      manager.seek(to: 0)
    }

    XCTAssertEqual(manager.currentPositiveText, "")
    XCTAssertEqual(manager.currentNegativeText, "")
    XCTAssertEqual(manager.currentEdits, 0)
  }

  func testSegmentationAtFiftyModifications() {
    let manager = onMain {
      TextHistoryManager(project: workspace, initialPositiveText: "")
    }

    onMain {
      for i in 0..<49 {
        manager.pushChange(
          range: NSRange(location: manager.currentPositiveText.utf16.count, length: 0),
          replacementText: String(i), textType: .positiveText)
      }
    }

    let expected = (0..<49).map(String.init).joined()
    XCTAssertEqual(manager.currentEdits, 50)
    XCTAssertEqual(manager.maxEdits, 50)
    XCTAssertEqual(manager.currentPositiveText, expected)

    onMain {
      manager.seek(to: 49)
    }

    let previousExpected = (0..<48).map(String.init).joined()
    XCTAssertEqual(manager.currentPositiveText, previousExpected)
    XCTAssertEqual(manager.currentEdits, 49)
  }

  func testUtf16RangeFullReplacementReplacesWholeText() {
    let emoji = "👨‍👩‍👧‍👦"
    let manager = onMain {
      TextHistoryManager(project: workspace, initialPositiveText: emoji)
    }

    onMain {
      manager.pushChange(
        range: NSRange(location: 0, length: manager.currentPositiveText.utf16.count),
        replacementText: "abc", textType: .positiveText)
    }

    XCTAssertEqual(manager.currentPositiveText, "abc")
  }

  func testUnsacredSeekPersistsResolvedLineage() {
    let replacementRange = TextRange(location: 0, length: 0)
    seedHistory(
      textHistories: [
        TextHistoryNode(
          lineage: 2, logicalTime: 0, startEdits: 0,
          startPositiveText: "", startNegativeText: "",
          modifications: [
            TextModification(type: .positiveText, range: replacementRange, text: "B")
          ]),
        TextHistoryNode(
          lineage: 3, logicalTime: 0, startEdits: 0,
          startPositiveText: "", startNegativeText: "",
          modifications: [
            TextModification(type: .positiveText, range: replacementRange, text: "A")
          ]),
      ],
      lineages: [
        TextLineageNode(lineage: 1, pointTo: 2)
      ])

    let manager = onMain {
      TextHistoryManager(project: workspace, initialPositiveText: "")
    }

    XCTAssertEqual(manager.lineage, 3)
    XCTAssertEqual(manager.currentPositiveText, "A")

    let isSacred = onMain {
      manager.seek(to: 1, lineage: 1)
    }
    XCTAssertFalse(isSacred)
    XCTAssertEqual(manager.lineage, 2)
    XCTAssertEqual(manager.currentPositiveText, "B")

    let storedLineage = workspace.dictionary["text_seek_to_lineage", Int.self]
    XCTAssertEqual(storedLineage, Int(manager.lineage))
  }

  func testInitializationSkipsOutOfBoundsPersistedModification() {
    seedHistory(
      textHistories: [
        TextHistoryNode(
          lineage: 1, logicalTime: 0, startEdits: 0,
          startPositiveText: "abc", startNegativeText: "",
          modifications: [
            TextModification(
              type: .positiveText, range: TextRange(location: 10, length: 3), text: "x")
          ])
      ])

    let manager = onMain {
      TextHistoryManager(project: workspace, initialPositiveText: "")
    }

    XCTAssertEqual(manager.currentPositiveText, "abc")
    XCTAssertEqual(manager.currentEdits, 1)
  }
}
