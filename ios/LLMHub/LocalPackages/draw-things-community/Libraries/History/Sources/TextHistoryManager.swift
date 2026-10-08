import Dflat
import Foundation

public final class TextHistoryManager {
  static let segmentationSize = 50  // This value cannot be modified as it is assumed since the beginning.

  private struct LogicalTimeAndLineage: Equatable & Hashable {
    var logicalTime: Int64
    var lineage: Int64
  }
  private struct InMemoryHistoryNode {
    var positiveText: String
    var negativeText: String
  }

  private static func normalizedRange(_ range: NSRange, in text: String) -> NSRange? {
    let textLength = (text as NSString).length
    guard range.location >= 0 && range.length >= 0 else { return nil }
    guard range.location <= textLength && range.length <= textLength - range.location else {
      return nil
    }
    return range
  }

  private static func normalizedRange(_ range: TextRange?, in text: String) -> NSRange? {
    guard
      let range = range,
      range.location >= 0,
      range.length >= 0
    else {
      return nil
    }
    return normalizedRange(
      NSRange(location: Int(range.location), length: Int(range.length)),
      in: text)
  }
  private var startPositiveText = ""
  private var startNegativeText = ""
  private var startEdits = 0
  public private(set) var currentPositiveText = ""
  public private(set) var currentNegativeText = ""
  private var history = [Int: InMemoryHistoryNode]()
  private var modifications = [TextModification]()
  public private(set) var maxEdits = 0
  public private(set) var currentEdits = 0
  private var maxLineage: Int64 = 0
  public private(set) var lineage: Int64 = 0
  private var maxLogicalTime: Int64 = 0
  private var logicalTime: Int64 = 0
  private var project: Workspace
  private var logicalVersion: Int = 0
  private var nodeLineageMap = [Int64: Int64]()  // old lineage -> new lineage map.
  private var nodeLineageCache = [LogicalTimeAndLineage: (TextHistoryNode, Int)]()  // Keyed by logical time and lineage.
  private var nodeCache = [Int64: (TextHistoryNode, Int)]()  // Keyed by logical time. Value is the node and a version integer.
  private var maxLogicalTimeForLineage = [Int64: Int64]()  // This never cleared up and will grow, but it is OK, because we probably at around 10k lineage or somewhere around that.
  public init(project: Workspace, initialPositiveText: String) {
    self.project = project
    let textHistories = project.fetch(for: TextHistoryNode.self).all(
      limit: .limit(1),
      orderBy: [TextHistoryNode.lineage.descending, TextHistoryNode.logicalTime.descending])
    guard let textHistory = textHistories.first else {
      startPositiveText = ""
      history[0] = InMemoryHistoryNode(positiveText: "", negativeText: "")
      currentPositiveText = initialPositiveText
      history[1] = InMemoryHistoryNode(positiveText: currentPositiveText, negativeText: "")
      modifications.append(
        TextModification(
          type: .positiveText, range: TextRange(location: 0, length: 0), text: currentPositiveText))
      maxEdits = 1
      currentEdits = 1
      return
    }
    // The current edits.
    setTextHistory(textHistory)
    maxLineage = lineage
    maxLogicalTime = logicalTime
    maxLogicalTimeForLineage[lineage] = logicalTime
    maxEdits = currentEdits
    if let seekTo = project.dictionary["text_seek_to", Int.self] {
      let _ = seek(
        to: seekTo, lineage: project.dictionary["text_seek_to_lineage", Int.self].map { Int64($0) })
    }
  }

  private func setTextHistory(_ textHistory: TextHistoryNode) {
    lineage = textHistory.lineage
    logicalTime = textHistory.logicalTime
    startEdits = Int(textHistory.startEdits)
    currentEdits = startEdits + textHistory.modifications.count
    modifications = textHistory.modifications
    startPositiveText = textHistory.startPositiveText ?? ""
    currentPositiveText = startPositiveText
    startNegativeText = textHistory.startNegativeText ?? ""
    currentNegativeText = startNegativeText
    history.removeAll()
    history[startEdits] = InMemoryHistoryNode(
      positiveText: currentPositiveText, negativeText: currentNegativeText)
    for (i, modification) in modifications.enumerated() {
      switch modification.type {
      case .positiveText:
        guard let range = Self.normalizedRange(modification.range, in: currentPositiveText) else {
          break
        }
        currentPositiveText = (currentPositiveText as NSString).replacingCharacters(
          in: range, with: modification.text ?? "")
      case .negativeText:
        guard let range = Self.normalizedRange(modification.range, in: currentNegativeText) else {
          break
        }
        currentNegativeText = (currentNegativeText as NSString).replacingCharacters(
          in: range, with: modification.text ?? "")
      }
      history[startEdits + i + 1] = InMemoryHistoryNode(
        positiveText: currentPositiveText, negativeText: currentNegativeText)
    }
  }

  private func saveCurrentText(completion: (() -> Void)? = nil) {
    let lineage = lineage
    let logicalTime = logicalTime
    let modifications = modifications
    let startPositiveText = startPositiveText
    let startNegativeText = startNegativeText
    let startEdits = startEdits
    project.performChanges([TextHistoryNode.self]) { transactionContext in
      let textHistory = TextHistoryNode(
        lineage: lineage, logicalTime: logicalTime, startEdits: Int64(startEdits),
        startPositiveText: startPositiveText, startNegativeText: startNegativeText,
        modifications: modifications)
      let upsertRequest = TextHistoryNodeChangeRequest.upsertRequest(textHistory)
      transactionContext.try(submit: upsertRequest)
    } completionHandler: { _ in
      DispatchQueue.main.async {
        completion?()
      }
    }
  }

  public func save() {
    // The only reason to save explicitly is because upon TextHistoryManager creation, we pre-filled,
    // and that can be identified by lineage == 0, logicalTime == 0, currentEdits = 1, startEdits = 0
    guard
      lineage == 0 && logicalTime == 0 && maxLineage == 0 && maxLogicalTime == 0 && startEdits == 0
        && currentEdits == 1 && maxEdits == 1
    else { return }
    saveCurrentText()
  }

  private func uniqueVersion() -> Int {
    let uniqueVersion = logicalVersion
    logicalVersion += 1
    return uniqueVersion
  }

  public func pushChange(range: NSRange, replacementText: String, textType: TextType) {
    dispatchPrecondition(condition: .onQueue(.main))
    switch textType {
    case .positiveText:
      guard Self.normalizedRange(range, in: currentPositiveText) != nil else {
        assertionFailure("Invalid range for text change.")
        return
      }
    case .negativeText:
      guard Self.normalizedRange(range, in: currentNegativeText) != nil else {
        assertionFailure("Invalid range for text change.")
        return
      }
    }
    // If we are not at the tip, when push a change, we need to rewind
    precondition(lineage <= maxLineage)
    if currentEdits != maxEdits || lineage < maxLineage {
      // We need to fork this edit otherwise we will end up remove some of the edits from the history.
      // If we are not the sacred lineage (as whether I am the maxLineage at this particular time)
      // Below: fetch the image history at this particular logical time from the sacred lineage.
      if let minLogicalTimeTextHistory =
        nodeCache[min(logicalTime, maxLogicalTime)]?.0
        ?? project.fetch(for: TextHistoryNode.self)
        .where(
          TextHistoryNode.logicalTime == min(logicalTime, maxLogicalTime),
          limit: .limit(1), orderBy: [TextHistoryNode.lineage.descending]
        ).first,  // This can be empty because it is 0, 0.
        // If is smaller than the sacred one.
        lineage < minLogicalTimeTextHistory.lineage
      {
        let textHistories = project.fetch(for: TextHistoryNode.self).where(
          TextHistoryNode.lineage == lineage)
        let newLineage = maxLineage + 1
        maxLineage = newLineage
        var textVersions = [Int]()
        for textHistory in textHistories {
          let uniqueVersion = uniqueVersion()
          textVersions.append(uniqueVersion)
          let node = (
            TextHistoryNode(
              lineage: newLineage, logicalTime: textHistory.logicalTime,
              startEdits: textHistory.startEdits,
              startPositiveText: textHistory.startPositiveText,
              startNegativeText: textHistory.startNegativeText,
              modifications: textHistory.modifications), uniqueVersion
          )
          nodeCache[textHistory.logicalTime] = node
          nodeLineageCache[
            LogicalTimeAndLineage(
              logicalTime: textHistory.logicalTime, lineage: newLineage)] = node
        }
        // Because we update lineage, but in the image history, the lineage pointing to is constant.
        // We need to keep track of our update both in memory, as well as on the disk.
        let lineage = lineage
        let project = project
        nodeLineageMap[lineage] = newLineage
        project.performChanges([TextHistoryNode.self, TextLineageNode.self]) { transactionContext in
          // If there are any lineages pointing to this one.
          let lineages = project.fetch(for: TextLineageNode.self).where(
            TextLineageNode.pointTo == lineage)
          for textHistory in textHistories {
            if let changeRequest = TextHistoryNodeChangeRequest.changeRequest(textHistory) {
              changeRequest.lineage = newLineage
              transactionContext.try(submit: changeRequest)
            }
          }
          let lineageNode = TextLineageNode(lineage: lineage, pointTo: newLineage)
          let lineageUpsertRequest = TextLineageNodeChangeRequest.upsertRequest(lineageNode)
          transactionContext.try(submit: lineageUpsertRequest)
          for lineage in lineages {
            if let changeRequest = TextLineageNodeChangeRequest.changeRequest(lineage) {
              changeRequest.pointTo = newLineage
              transactionContext.try(submit: changeRequest)
            }
          }
        } completionHandler: { _ in
          DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            for (i, textHistory) in textHistories.enumerated() {
              if let node = self.nodeCache[textHistory.logicalTime], node.1 == textVersions[i] {
                self.nodeCache[textHistory.logicalTime] = nil
              }
              let logicalTimeAndLineage = LogicalTimeAndLineage(
                logicalTime: textHistory.logicalTime, lineage: newLineage)
              if let node = self.nodeLineageCache[logicalTimeAndLineage], node.1 == textVersions[i]
              {
                self.nodeLineageCache[logicalTimeAndLineage] = nil
              }
            }
          }
        }
      }
      lineage = maxLineage + 1
      maxLineage = lineage
      maxLogicalTime = logicalTime
      maxLogicalTimeForLineage[lineage] = maxLogicalTime
      maxEdits = startEdits + modifications.count  // Reset max edit to the fork point, in case it is way larger.
      if currentEdits < maxEdits {
        modifications.removeLast(maxEdits - currentEdits)
        for i in (currentEdits + 1)...maxEdits {
          history[i] = nil
        }
      }
      maxEdits = currentEdits
      // I don't need to save current text here even if the lineage changed, because:
      // There are two scenarioes, whether I move forward to the next logical time, or not.
      // If I don't, it is OK, because I will save the current state + new modification. So for case
      // we actually move logical time forward, we need to know whether this materially change the
      // current state. It is actually not. Because we are still on a good lineage (the lineage is updated
      // in above code if it is not a good one), and will be selected as the previous lineage because
      // it is oldest.
    }
    currentEdits += 1
    maxEdits = currentEdits
    modifications.append(
      TextModification(
        type: textType,
        range: TextRange(location: Int32(range.location), length: Int32(range.length)),
        text: replacementText))
    switch textType {
    case .positiveText:
      let newText = (currentPositiveText as NSString).replacingCharacters(
        in: range, with: replacementText)
      history[currentEdits] = InMemoryHistoryNode(
        positiveText: newText, negativeText: currentNegativeText)
      self.currentPositiveText = newText
    case .negativeText:
      let newText = (currentNegativeText as NSString).replacingCharacters(
        in: range, with: replacementText)
      history[currentEdits] = InMemoryHistoryNode(
        positiveText: currentPositiveText, negativeText: newText)
      self.currentNegativeText = newText
    }
    let textVersion = uniqueVersion()
    project.dictionary["text_seek_to", Int.self] = nil
    guard modifications.count > Self.segmentationSize - 1 else {
      let node = (
        TextHistoryNode(
          lineage: lineage, logicalTime: logicalTime, startEdits: Int64(startEdits),
          startPositiveText: startPositiveText, startNegativeText: startNegativeText,
          modifications: modifications), textVersion
      )
      nodeCache[logicalTime] = node
      let logicalTimeAndLineage = LogicalTimeAndLineage(logicalTime: logicalTime, lineage: lineage)
      nodeLineageCache[logicalTimeAndLineage] = node
      let logicalTime = logicalTime
      saveCurrentText {
        DispatchQueue.main.async { [weak self] in
          guard let self = self else { return }
          // It is saved, remove.
          if let node = self.nodeCache[logicalTime], node.1 == textVersion {
            self.nodeCache[logicalTime] = nil
          }
          if let node = self.nodeLineageCache[logicalTimeAndLineage], node.1 == textVersion {
            self.nodeLineageCache[logicalTimeAndLineage] = nil
          }
        }
      }
      return
    }
    precondition(modifications.count == Self.segmentationSize)
    startEdits = currentEdits
    startPositiveText = currentPositiveText
    startNegativeText = currentNegativeText
    logicalTime += 1
    maxLogicalTime = logicalTime
    maxLogicalTimeForLineage[lineage] = maxLogicalTime
    modifications.removeAll()
    let node = (
      TextHistoryNode(
        lineage: lineage, logicalTime: logicalTime, startEdits: Int64(startEdits),
        startPositiveText: startPositiveText, startNegativeText: startNegativeText,
        modifications: modifications), textVersion
    )
    nodeCache[logicalTime] = node
    let logicalTimeAndLineage = LogicalTimeAndLineage(logicalTime: logicalTime, lineage: lineage)
    nodeLineageCache[logicalTimeAndLineage] = node
    let logicalTime = logicalTime
    saveCurrentText {
      DispatchQueue.main.async { [weak self] in
        guard let self = self else { return }
        // It is saved, remove.
        if let node = self.nodeCache[logicalTime], node.1 == textVersion {
          self.nodeCache[logicalTime] = nil
        }
        if let node = self.nodeLineageCache[logicalTimeAndLineage], node.1 == textVersion {
          self.nodeLineageCache[logicalTimeAndLineage] = nil
        }
      }
    }
  }

  public func saveCursor() {
    guard currentEdits <= maxEdits else { return }
    project.dictionary["text_seek_to"] = currentEdits
  }

  public func seek(to edits: Int) {
    guard edits != self.currentEdits else {
      return
    }
    let _ = seek(to: edits, lineage: nil)
  }

  // Return whether we are on the sacred lineage, which is the lineage can be represented by the slider.
  public func seek(to edits: Int, lineage: Int64?) -> Bool {
    dispatchPrecondition(condition: .onQueue(.main))
    var edits = edits
    if lineage == nil {
      assert(edits >= 0 && edits <= maxEdits)
      edits = min(max(edits, 0), maxEdits)
    }
    var isSacred = true
    if edits < startEdits || edits > startEdits + modifications.count
      || (lineage != nil && lineage! < maxLineage)
    {
      // Since each segment is at segment size, we can find the startEdits by compute the logicalTime = edits / segmentationSize.
      let editLogicalTime = Int64(edits / Self.segmentationSize)
      // First check cache (newest, haven't written to disk yet), and then check disk.
      guard
        let textHistory =
          nodeCache[editLogicalTime]?.0
          ?? project.fetch(for: TextHistoryNode.self).where(
            TextHistoryNode.logicalTime == editLogicalTime, limit: .limit(1),
            orderBy: [TextHistoryNode.lineage.descending]
          ).first
      else { return false }
      let upToDateLineage = lineage.map {
        var previousUpToDateLineage: Int64? = nil
        var upToDateLineage = nodeLineageMap[$0]
        while upToDateLineage != nil {
          previousUpToDateLineage = upToDateLineage
          upToDateLineage = nodeLineageMap[upToDateLineage!]
        }
        if let previousUpToDateLineage = previousUpToDateLineage {
          return previousUpToDateLineage
        }
        if let lineageNode = project.fetch(for: TextLineageNode.self).where(
          TextLineageNode.lineage == $0
        ).first {
          nodeLineageMap[$0] = lineageNode.pointTo
          return lineageNode.pointTo
        }
        return $0
      }
      if let lineage = upToDateLineage, textHistory.lineage != lineage {
        guard
          let oldTextHistory =
            nodeLineageCache[LogicalTimeAndLineage(logicalTime: editLogicalTime, lineage: lineage)]?
            .0
            ?? project.fetch(for: TextHistoryNode.self).where(
              TextHistoryNode.logicalTime == editLogicalTime && TextHistoryNode.lineage == lineage,
              limit: .limit(1)
            ).first
        else { return false }
        // If the old text history (text history on the older lineage) is shorter / same as the sacred text history, as well as it does prefix matching, we treat this as sacred.
        let modificationsUpToRequestedEdits = edits - Int(textHistory.startEdits)
        if oldTextHistory.startEdits == textHistory.startEdits
          && oldTextHistory.startPositiveText == textHistory.startPositiveText
          && oldTextHistory.startNegativeText == textHistory.startNegativeText
          && modificationsUpToRequestedEdits <= oldTextHistory.modifications.count
          && textHistory.modifications.prefix(modificationsUpToRequestedEdits)
            == oldTextHistory.modifications.prefix(modificationsUpToRequestedEdits)
        {
          setTextHistory(textHistory)
        } else {
          setTextHistory(oldTextHistory)
          isSacred = false  // This is not the most up to date one.
        }
        if let maxLogicalTime = maxLogicalTimeForLineage[lineage] {
          self.maxLogicalTime = maxLogicalTime
        } else {
          let oldMaxLogicalTimeTextHistory = project.fetch(for: TextHistoryNode.self).where(
            TextHistoryNode.lineage == lineage,
            limit: .limit(1), orderBy: [TextHistoryNode.logicalTime.descending]
          ).first!
          maxLogicalTime = oldMaxLogicalTimeTextHistory.logicalTime
          maxLogicalTimeForLineage[lineage] = maxLogicalTime
        }
      } else {
        setTextHistory(textHistory)
        // Even if lineage matches the requested, we may not be on the sacred lineage because the
        // sacred lineage is shorter. Checking if the requested logicalTime is smaller than maxLogicalTime.
        if editLogicalTime > maxLogicalTime {
          isSacred = false
        }
      }
    }
    // Now we end up with a easy situation, we just find the ones from history.
    assert(edits >= startEdits && edits <= startEdits + modifications.count)
    currentEdits = min(max(edits, startEdits), startEdits + modifications.count)
    currentPositiveText = history[currentEdits]!.positiveText
    currentNegativeText = history[currentEdits]!.negativeText
    // Only update where to seek to if it is on sacred lineage.
    if isSacred {
      project.dictionary["text_seek_to"] = edits
      project.dictionary["text_seek_to_lineage", Int.self] = nil
    } else {
      project.dictionary["text_seek_to"] = edits
      if let _ = lineage {
        // Persist the resolved lineage we actually landed on (not the requested alias lineage).
        project.dictionary["text_seek_to_lineage", Int.self] = Int(self.lineage)
      } else {
        project.dictionary["text_seek_to_lineage", Int.self] = nil
      }
    }
    return isSacred
  }
}
