import Combine
import DataModels
import Diffusion
import Foundation
import ImageSegmentation
import JavaScriptCore
import NNC
import ScriptDataModels
import Utils
import Vision

final class MaskManager {
  // Start actual handles at 1, because 0 indicates that a null mask handle was passed from JS
  let nullMask = JSMask(handle: 0)
  private var nextMaskHandle: Int = 1
  private var maskHandleToMask: [Int: Tensor<UInt8>] = [:]

  func mask(forJSMask jsMask: JSMask) -> Tensor<UInt8>? {
    return maskHandleToMask[jsMask.handle]
  }

  func setMask(_ mask: Tensor<UInt8>, forJSMask jsMask: JSMask) {
    maskHandleToMask[jsMask.handle] = mask
  }

  func createNewMask() -> JSMask {
    let mask = JSMask(handle: nextMaskHandle)
    nextMaskHandle += 1
    return mask
  }
}

final class JSMask: Codable {
  let handle: Int

  fileprivate init(handle: Int) {
    self.handle = handle
  }
}

public func jsonObject(from object: Encodable) throws -> Any {
  let data = try JSONEncoder().encode(object)
  return try JSONSerialization.jsonObject(
    with: data, options: [.mutableContainers, .mutableLeaves])
}

public func swiftObject<T: Decodable>(jsonObject: Any, type: T.Type) throws -> T {
  let data = try JSONSerialization.data(withJSONObject: jsonObject)
  return try JSONDecoder().decode(type, from: data)
}

public enum ScriptExecutionState {
  case running
  case cancelled
  case ended
}

public enum OpenPurpose: String {
  case image
  case mask
  case depthMap
  case extractDepthMap
  case scribble
  case pose
  case color
  case custom
  case shuffle
  case addImage
}

public enum ModelType: String {
  case LoRA = "LoRA"
  case ControlNet = "Control Net"
  case notAvailable = "N/A"
}

public enum SupportedPrefix: String {
  case base64Png = "data:image/png;base64,"
  case base64Jpg = "data:image/jpeg;base64,"
  case base64Webp = "data:image/webp;base64,"
  case file = "file://"
  case notAvailable = "N/A"
}

extension SupportedPrefix {
  public static func supportedPrefix(from srcContent: String) -> (SupportedPrefix, String) {
    if srcContent.hasPrefix(SupportedPrefix.base64Png.rawValue) {
      return (
        .base64Png,
        srcContent.replacingOccurrences(of: SupportedPrefix.base64Png.rawValue, with: "")
      )
    }
    if srcContent.hasPrefix(SupportedPrefix.base64Jpg.rawValue) {
      return (
        .base64Jpg,
        srcContent.replacingOccurrences(of: SupportedPrefix.base64Jpg.rawValue, with: "")
      )
    }
    if srcContent.hasPrefix(SupportedPrefix.base64Webp.rawValue) {
      return (
        .base64Webp,
        srcContent.replacingOccurrences(of: SupportedPrefix.base64Webp.rawValue, with: "")
      )
    }
    if srcContent.hasPrefix(SupportedPrefix.file.rawValue) {
      return (.file, srcContent.replacingOccurrences(of: SupportedPrefix.file.rawValue, with: ""))
    }
    return (.notAvailable, "")
  }
}

@objc protocol JSInterop: JSExport {
  func repl()
  func log(_ message: String, _ type: Int)
  func generateImage(_ args: [String: Any])
  func createControl(_ name: String) -> [String: Any]?
  func createLoRA(_ name: String) -> [String: Any]?
  func CLIP(_ texts: [String]) -> [Float]
  func fillMaskRectangle(
    _ maskDictionary: [String: Any], _ rect: CGRect, _ value: UInt8)
  func moveCanvas(_ x: Double, _ y: Double)
  func updateCanvasSize(_ configurationDictionary: [String: Any])
  func createMask(_ width: Double, _ height: Double, _ value: UInt8) -> Int
  func createForegroundMask() -> Int
  func createBackgroundMask() -> Int
  func createBodyMask(_ types: [String], _ extraArea: Bool) -> Int
  func setCanvasZoom(_ zoomScale: Double)
  func canvasZoom() -> Double
  func existingConfiguration() -> [String: Any]
  func existingPrompts() -> [String: String]
  func currentMask() -> [String: Any]?
  func topLeftCorner() -> [String: Any]
  func boundingBox() -> CGRect
  func clearCanvas()
  func listFilesWithinDirectory(_ directory: String) -> [String]
  func listFilesUnderPicturesWithinDirectory(_ directory: String) -> [String]
  func listFilesUnderPictures() -> [String]
  func picturesPath() -> String?
  func loadImageFileToCanvas(_ file: String)
  func loadImageSrcToCanvas(_ srcContent: String)
  func imageSizeFromSrc(_ srcContent: String) -> CGSize
  func saveImageFileFromCanvas(_ file: String, _ visibleRegionOnly: Bool)
  func saveImageSrcFromCanvas(_ visibleRegionOnly: Bool) -> String?
  func loadLayerFromPhotos(_ type: String)
  func loadLayerFromFiles(_ type: String)
  func loadLayerFromSrc(_ srcContent: String, _ type: String)
  func loadLayerFromJson(_ json: String, _ type: String)
  func saveLayerSrc(_ type: String) -> String?
  func maskSrc(_ maskJson: [String: Any]) -> String?
  func clearMoodboard()
  func extractDepthMap()
  func removeFromMoodboardAt(_ index: Int)
  func setMoodboardImageWeight(_ weight: Double, _ index: Int)
  func detectFaces() -> [[String: Any]]
  func detectHands() -> [[String: Any]]
  func downloadBuiltins(_ files: [String])
  func areModelsDownloaded(_ files: [String]) -> [Bool]
  func requestFromUser(_ title: String, _ confirm: String, _ config: [[String: Any]]) -> [Any]
  func screenSize() -> [String: Any]
  func answer(_: String, _: String) -> String
}

public protocol ScriptExecutorDelegate: AnyObject {
  func createControl(_ name: String) throws -> Control
  func createLoRA(_ name: String) throws -> LoRA
  func topLeftCorner() -> CGPoint
  func currentMask() -> Tensor<UInt8>?
  func foregroundMask() -> Tensor<UInt8>?
  func backgroundMask() -> Tensor<UInt8>?
  func createBodyMask(_ categories: [SCHPMaskGenerator.Category], _ extraArea: Bool) -> Tensor<
    UInt8
  >?
  func maskImageData(_ mask: Tensor<UInt8>) -> Data?
  func generateImage(
    prompt: String?, negativePrompt: String?, configuration: GenerationConfiguration,
    mask: Tensor<UInt8>?
  ) -> Bool
  func setZoomScale(_ zoomScale: CGFloat)
  func zoomScale() -> CGFloat
  func updateCanvasSize(_ configuration: GenerationConfiguration)
  func moveCanvas(x: Int, y: Int)
  func existingConfiguration() -> GenerationConfiguration
  func existingPrompts() -> (prompt: String, negativePrompt: String)
  func boundingBox() -> CGRect
  func log(_ message: String, type: Int)
  func logException(_ message: String, line: Int, stack: String)
  func logJavascript(title: String, script: String, index: Int)
  func logMask(_ mask: Tensor<UInt8>)
  func openREPLConsole()
  func evaluateScriptGroupBegan()
  func evaluateScriptBegan(sessionId: UInt64)
  func evaluateScriptBeforeREPL()
  func evaluateScriptEnded(
    session: ScriptExecutionSession, models: [String], loras: [String], controls: [String])
  func evaluateScriptGroupEnded()
  func clearCanvas()
  func CLIP(_ texts: [String]) throws -> [Float]
  func loadImageFileToCanvas(_ file: String)
  func loadImageSrcToCanvas(_ srcContent: String) throws
  func imageSizeFromSrc(_ srcContent: String) -> CGSize
  func saveImageFileFromCanvas(_ file: String, _ visibleRegionOnly: Bool)
  func saveImageDataFromCanvas(_ visibleRegionOnly: Bool) -> Data?
  func loadLayerFromPhotos(type: String) throws
  func loadLayerFromFiles(type: String) throws
  func loadLayerFromSrc(_ srcContent: String, type: String) throws
  func loadLayerFromJson(_ json: String, type: String) throws
  func saveLayerData(type: String) throws -> Data?
  func clearMoodboard() throws
  func extractDepthMap() throws
  func removeFromMoodboardAt(_ index: Int) throws
  func setMoodboardImageWeight(_ weight: Double, _ index: Int) throws
  func detectFaces() throws -> [CGRect]
  func detectHands() throws -> [CGRect]
  func downloadBuiltins(_ files: [String]) throws
  func areModelsDownloaded(_ files: [String]) throws -> [Bool]
  func requestFromUser(title: String, confirm: String, _ config: [[String: Any]]) throws -> (
    lifetimeObjects: [AnyObject], result: [Any]
  )
  func screenSize() -> [String: Any]
  func answer(model: String, question: String) throws -> String
}

public struct ScriptExecutionSession {
  // higher 44-bit is random, lower 20-bit is seconds since 1970.
  // Usually it is already good with random number because SystemRNG
  // uses high quality source. 20-bit is enough for a year wrap-around.
  // So we have unique session id every second for a year, and if it happens
  // in a second, we have enough entropy from rng to differentiate them.
  // set the first bit to 0 to avoid UInt -> Int convertion overflow
  public let sessionId: UInt64
  public let sessionGroupId: UInt64
  public let file: String
}

@objc public final class ScriptExecutor: NSObject {
  enum REPLCommand {
    case exit
    case javascript(file: String, script: String, completionHandler: (Bool) -> Void)
  }
  fileprivate static var systemRandomNumberGenerator = SystemRandomNumberGenerator()
  private static let queue = DispatchQueue(label: "com.draw-things.script", qos: .userInteractive)
  var hasExecuted = false
  var hasCancelled = false
  var hasException = false
  var lifetimeObjects = [AnyObject]()
  let maskManager = MaskManager()
  weak var delegate: ScriptExecutorDelegate?
  public let scripts: [(file: String, script: String)]
  var context: JSContext?
  public let isREPL: Bool
  private let replLock: DispatchQueue
  private let replSignal: DispatchSemaphore
  private var replCommand: [REPLCommand]
  private var models: Set<String> = []
  private var loras: Set<String> = []
  private var controls: Set<String> = []

  private let subject: PassthroughSubject<ScriptExecutionState, Never> =
    PassthroughSubject<ScriptExecutionState, Never>()
  public var executionStatePublisher: AnyPublisher<ScriptExecutionState, Never> {
    return subject.eraseToAnyPublisher()
  }

  public init(
    scripts: [(file: String, script: String)], delegate: ScriptExecutorDelegate, isREPL: Bool
  ) {
    self.scripts = scripts
    self.delegate = delegate
    self.isREPL = isREPL
    replLock = DispatchQueue(label: "com.draw-things.repl")
    replSignal = DispatchSemaphore(value: 0)
    replCommand = []
  }
}

extension ScriptExecutor: JSInterop {

  public enum CancellationError: Error {
    case cancelled
  }

  func forwardExceptionsToJS<T: DefaultCreatable>(_ block: () throws -> T) -> T {
    do {
      return try block()
    } catch CancellationError.cancelled {
      context?.exception = JSValue(newErrorFromMessage: "cancelled", in: context)
      hasCancelled = true
      return T.defaultInstance()
    } catch let error {
      context?.exception = JSValue(newErrorFromMessage: "\(error)", in: context)
      return T.defaultInstance()
    }
  }

  func forwardExceptionsToJS(_ block: () throws -> Void) {
    let _ = forwardExceptionsToJS {
      try block()
      return 0
    }
  }

  func updateCanvasSize(_ configurationDictionary: [String: Any]) {
    forwardExceptionsToJS {
      let configuration = try swiftObject(
        jsonObject: configurationDictionary, type: JSGenerationConfiguration.self
      ).createGenerationConfiguration()
      delegate?.updateCanvasSize(configuration)
    }
  }

  func fixDimensionIfNecessary(_ dimension: inout UInt32) {
    if dimension % 64 != 0 {
      log(
        "All dimensions (width, height, hi-res fix width, hi-res fix height must be divisible by 64. Rounding \(dimension) down to \(dimension - dimension % 64)..."
      )
      dimension -= dimension % 64
    }
  }

  func generateImage(_ args: [String: Any]) {  // configurationDictionary: [String: Any], _ maskDictionary: [String: Any]?) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let mask: Tensor<UInt8>?
      if let maskDictionary = args["mask"] as? [String: Any] {
        let jsMask = try swiftObject(jsonObject: maskDictionary, type: JSMask.self)
        guard let storedMask = maskManager.mask(forJSMask: jsMask) else {
          throw "Unrecognized handle: \(jsMask.handle)"
        }
        mask = storedMask
      } else {
        mask = nil
      }
      guard let argsConfiguration = args["configuration"] as? [String: Any] else {
        throw "Missing configuration parameter"
      }
      let jsConfiguration = try swiftObject(
        jsonObject: argsConfiguration, type: JSGenerationConfiguration.self)
      // Do additional check to report most likely any misspells.
      let mirror = Mirror(reflecting: jsConfiguration)
      let existingKeys = Set(
        mirror.children.compactMap {
          $0.label
        })
      for key in argsConfiguration.keys {
        guard !existingKeys.contains(key) else { continue }
        log("pipeline.run() - configuration contains unknown key \(key)")
      }

      fixDimensionIfNecessary(&jsConfiguration.width)
      fixDimensionIfNecessary(&jsConfiguration.height)
      fixDimensionIfNecessary(&jsConfiguration.hiresFixWidth)
      fixDimensionIfNecessary(&jsConfiguration.hiresFixHeight)
      fixDimensionIfNecessary(&jsConfiguration.decodingTileWidth)
      fixDimensionIfNecessary(&jsConfiguration.decodingTileHeight)
      fixDimensionIfNecessary(&jsConfiguration.decodingTileOverlap)
      let configuration = jsConfiguration.createGenerationConfiguration()
      if let model = configuration.model {
        models.insert(model)
      }
      configuration.loras.forEach {
        if let file = $0.file {
          loras.insert(file)
        }
      }
      configuration.controls.forEach {
        if let file = $0.file {
          controls.insert(file)
        }
      }
      let prompt = (args["prompt"] as? NSString) as? String
      let negativePrompt = (args["negativePrompt"] as? NSString) as? String
      let notAborted = delegate.generateImage(
        prompt: prompt, negativePrompt: negativePrompt, configuration: configuration, mask: mask)
      if !notAborted {
        throw CancellationError.cancelled
      }
    }
  }

  func markAsOutput() {

  }

  func setCanvasZoom(_ zoomScale: Double) {
    if (zoomScale * 60).rounded() != zoomScale * 60 {
      log("Warning: setCanvasZoom parameter should be divisible by 60")
    }
    delegate?.setZoomScale(CGFloat(zoomScale))
  }

  func canvasZoom() -> Double {
    return (delegate?.zoomScale()).map({ Double($0) }) ?? 1
  }

  func moveCanvas(_ x: Double, _ y: Double) {
    if x != x.rounded() || y != y.rounded() {
      log("Warning: x and y coordinates to moveCanvas should be integers")
    }
    delegate?.moveCanvas(x: Int(x), y: Int(y))
  }

  func fillMaskRectangle(
    _ maskDictionary: [String: Any], _ rect: CGRect, _ value: UInt8
  ) {
    forwardExceptionsToJS {
      let jsMask = try swiftObject(jsonObject: maskDictionary, type: JSMask.self)
      guard var mask = maskManager.mask(forJSMask: jsMask) else {
        throw "Unrecognized handle: \(jsMask.handle)"
      }
      let xStart = Int(rect.origin.x)
      let yStart = Int(rect.origin.y)
      let yEnd = yStart + Int(rect.size.height)
      let xEnd = xStart + Int(rect.size.width)
      guard yEnd <= mask.shape[0] else {
        throw "Mask y larger than mask dimension: y \(yEnd), dimension: \(mask.shape[0])"
      }
      guard xEnd <= mask.shape[1] else {
        throw "Mask x larger than mask dimension: y \(xEnd), dimension: \(mask.shape[1])"
      }
      guard xStart >= 0 else {
        throw "Mask x must be greater than or equal to 0: \(xStart)"
      }
      guard yStart >= 0 else {
        throw "Mask y must be greater than or equal to 0: \(yStart))"
      }
      for y in yStart..<yEnd {
        for x in xStart..<xEnd {
          mask[y, x] = value
        }
      }
      maskManager.setMask(mask, forJSMask: jsMask)
    }
  }

  // console.log is variadic, but we only handle the first string
  func log(_ message: String, _ type: Int) {
    print("JavaScriptCore: \(message)")
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      delegate.log(message, type: type)
    }
  }

  func log(_ message: String) {
    log(message, 0)
  }

  func createLoRA(_ name: String) -> [String: Any]? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let lora = try delegate.createLoRA(name)
      let jsLoRA = JSLoRA(lora: lora)
      let data = try JSONEncoder().encode(jsLoRA)
      return ((try JSONSerialization.jsonObject(with: data)) as! [String: Any])
    }
  }

  func clearCanvas() {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      delegate.clearCanvas()
    }
  }

  func CLIP(_ texts: [String]) -> [Float] {
    return forwardExceptionsToJS {
      guard let delegate else { throw "No delegate" }
      return try delegate.CLIP(texts)
    }
  }

  func createControl(_ name: String) -> [String: Any]? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let control = try delegate.createControl(name)
      let jsControl = JSControl(control: control)
      let data = try JSONEncoder().encode(jsControl)
      return ((try JSONSerialization.jsonObject(with: data)) as! [String: Any])
    }
  }

  func createMask(_ width: Double, _ height: Double, _ value: UInt8) -> Int {
    return forwardExceptionsToJS {
      if width != width.rounded() || height != height.rounded() {
        throw "Width and height must be integers"
      }
      var mask = Tensor<UInt8>(.CPU, .NC(Int(height), Int(width)))
      for y in 0..<Int(height) {
        for x in 0..<Int(width) {
          mask[y, x] = value
        }
      }
      let jsMask = maskManager.createNewMask()
      maskManager.setMask(mask, forJSMask: jsMask)
      // Just return the handle and not the whole mask object, because in JS,
      // `new Mask(handle)` is different from `{"handle": handle}` (with the latter, `mask.func(...)` won't work)
      return jsMask.handle
    }
  }

  func createForegroundMask() -> Int {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      guard let foregroundMask = delegate.foregroundMask() else { return 0 }
      let jsMask = maskManager.createNewMask()
      maskManager.setMask(foregroundMask, forJSMask: jsMask)
      return jsMask.handle
    }
  }

  func createBackgroundMask() -> Int {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      guard let backgroundMask = delegate.backgroundMask() else { return 0 }
      let jsMask = maskManager.createNewMask()
      maskManager.setMask(backgroundMask, forJSMask: jsMask)
      return jsMask.handle
    }
  }

  func answer(_ model: String, _ question: String) -> String {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      return try delegate.answer(model: model, question: question)
    }
  }

  func createBodyMask(_ types: [String], _ extraArea: Bool) -> Int {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let categories = types.compactMap {
        if $0.lowercased().hasPrefix("lower") {
          return SCHPMaskGenerator.Category.lowerBody
        } else if $0.lowercased().hasPrefix("upper") {
          return SCHPMaskGenerator.Category.upperBody
        } else if $0.lowercased().hasPrefix("dress") || $0.lowercased().hasPrefix("clothing") {
          return SCHPMaskGenerator.Category.dresses
        } else if $0.lowercased().hasPrefix("neck") {
          return SCHPMaskGenerator.Category.neck
        }
        return nil
      }
      guard !categories.isEmpty else { throw "No valid body category" }
      guard let bodyMask = delegate.createBodyMask(categories, extraArea) else { return 0 }
      let jsMask = maskManager.createNewMask()
      maskManager.setMask(bodyMask, forJSMask: jsMask)
      return jsMask.handle
    }
  }

  func maskSrc(_ maskJson: [String: Any]) -> String? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let jsMask = try swiftObject(jsonObject: maskJson, type: JSMask.self)
      guard let storedMask = self.maskManager.mask(forJSMask: jsMask) else {
        throw "Unrecognized handle: \(jsMask.handle)"
      }

      guard let data = delegate.maskImageData(storedMask) else { return nil }
      return SupportedPrefix.base64Png.rawValue + data.base64EncodedString()
    }
  }

  func topLeftCorner() -> [String: Any] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let point = delegate.topLeftCorner()
      let jsPoint = JSPoint(point: point)
      return try jsonObject(from: jsPoint) as! [String: Any]
    }
  }

  func boundingBox() -> CGRect {
    guard let delegate = delegate else { return .zero }
    return delegate.boundingBox()
  }

  func currentMask() -> [String: Any]? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      guard let mask = delegate.currentMask() else {
        return nil
      }
      let jsMask = maskManager.createNewMask()
      maskManager.setMask(mask, forJSMask: jsMask)
      return try jsonObject(from: jsMask) as! [String: Any]?
    }
  }

  func handleException(value: JSValue?) {
    let message = value?.objectForKeyedSubscript("message")?.toString() ?? ""
    let lineNumber = Int(value?.objectForKeyedSubscript("line")?.toInt32() ?? -1)
    let stackString = value?.objectForKeyedSubscript("stack")?.toString() ?? ""
    delegate?.logException(message, line: lineNumber, stack: stackString)
    hasException = true
  }

  public func run(_ times: Int = 1) {
    Self.queue.async {
      if self.hasExecuted {
        // It's one-shot, e.g. to avoid having to manually let go of all masks stored in maskManager
        fatalError(
          "This object is one-shot, create another instance if you need to run execute() again")
      }
      self.context = JSContext()
      self.hasExecuted = true
      self.subject.send(.running)
      self.context?.exceptionHandler = { [weak self] context, value in
        guard let self = self else { return }
        // If it is not cancellation error, we handle it.
        guard !self.hasCancelled else {
          self.subject.send(.cancelled)
          return
        }
        self.handleException(value: value)
        self.subject.send(.ended)
      }
      self.context?.setObject(self, forKeyedSubscript: "__dtHooks" as NSCopying & NSObjectProtocol)
      self.delegate?.evaluateScriptGroupBegan()
      // It's good to evaluate these as two separate scripts, rather than pasting them together into one script,
      // so that stack trace line numbers are accurate for the user for errors in their code
      self.context?.evaluateScript(Self.SharedScript)
      self.delegate?.logJavascript(
        title: "Shared.js", script: Self.SharedScript, index: 0)
      let firstSessionId = ScriptExecutionSession.sessionId()
      for (i, script) in self.scripts.enumerated() {
        let session =
          i == 0
          ? ScriptExecutionSession(
            sessionId: firstSessionId, sessionGroupId: firstSessionId, file: script.file)
          : ScriptExecutionSession(
            sessionId: ScriptExecutionSession.sessionId(), sessionGroupId: firstSessionId,
            file: script.file)
        self.delegate?.evaluateScriptBegan(sessionId: session.sessionId)
        self.delegate?.logJavascript(
          title: (script.file as NSString).lastPathComponent, script: script.script, index: i + 1)
        self.context?.evaluateScript(
          """
          (function (){
          \(script.script)
          })();
          """)
        self.delegate?.evaluateScriptEnded(
          session: session, models: self.models.sorted(), loras: self.loras.sorted(),
          controls: self.controls.sorted())
        if self.hasCancelled || self.hasException {
          break
        }
      }
      self.hasException = false
      self.hasCancelled = false
      if self.isREPL {
        self.delegate?.evaluateScriptBeforeREPL()
        self.replLoop()
      }
      self.context = nil  // In case this will help prevent a retain cycle
      self.lifetimeObjects.removeAll()
      self.delegate?.evaluateScriptGroupEnded()
      self.subject.send(.ended)
    }
  }

  private func replLoop() {
    dispatchPrecondition(condition: .onQueue(Self.queue))
    // Now wait for signal.
    guard let context = context else { return }
    while true {
      replSignal.wait()
      let command = replLock.sync {
        replCommand.removeFirst()
      }
      switch command {
      case .exit:  // Cleanup all remaining executions.
        replLock.sync {
          let remainingCount = replCommand.count
          replCommand = []
          // Balance out the semaphore.
          for _ in 0..<remainingCount {
            replSignal.wait()
          }
        }
        return
      case .javascript(_, let script, let completionHandler):
        if let result = context.evaluateScript(script) {
          if result.isObject || result.isArray, let object = result.toObject() {
            // Inspect whether it is some types we know of:

            if let maskDictionary = object as? [String: Any],
              let jsMask = try? swiftObject(jsonObject: maskDictionary, type: JSMask.self),
              let storedMask = maskManager.mask(forJSMask: jsMask)
            {
              delegate?.logMask(storedMask)
            } else if let data = try? JSONSerialization.data(
              withJSONObject: object, options: .prettyPrinted),
              let result = String(data: data, encoding: .utf8)
            {
              delegate?.log(result, type: 0)
            } else {
              delegate?.log(result.toString(), type: 0)
            }
          } else if !result.isUndefined {
            delegate?.log(result.toString(), type: 0)
          }
        }
        completionHandler(!hasException)
        hasException = false
        hasCancelled = false
      }
    }
  }

  public func abort() {
    replLock.sync {
      replCommand.append(.exit)
      replSignal.signal()
    }
  }

  public func append(file: String, script: String, completionHandler: @escaping (Bool) -> Void) {
    replLock.sync {
      replCommand.append(
        .javascript(file: file, script: script, completionHandler: completionHandler))
      replSignal.signal()
    }
  }

  func repl() {
    // If already is in REPL mode, no need to trigger the loop.
    guard !isREPL else { return }
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      delegate.openREPLConsole()
    }
    replLoop()
  }

  // TODO: should we add "Copy" to the end of this method so people know that mutating it won't change the existing one?
  func existingConfiguration() -> [String: Any] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let configuration = delegate.existingConfiguration()
      let jsConfiguration = JSGenerationConfiguration(configuration: configuration)
      return try jsonObject(from: jsConfiguration) as! [String: Any]
    }
  }

  public static func ensurePicturesDirectoryExists() {
    #if targetEnvironment(macCatalyst)
      return
    #else
      let fileManager = FileManager.default
      guard let documentUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
      else { return }
      let systemPicturesUrl = documentUrl.appendingPathComponent("Pictures")
      try? FileManager.default.createDirectory(
        at: systemPicturesUrl, withIntermediateDirectories: true)
    #endif
  }

  func listFilesWithinDirectory(_ directory: String) -> [String] {
    return forwardExceptionsToJS { () -> [String] in
      let fileManager = FileManager.default
      let directoryUrl = URL(fileURLWithPath: directory)
      let fileUrls = try fileManager.contentsOfDirectory(
        at: directoryUrl, includingPropertiesForKeys: nil)
      return fileUrls.map { $0.path }
    }
  }

  func listFilesUnderPicturesWithinDirectory(_ directory: String) -> [String] {
    return forwardExceptionsToJS { () -> [String] in
      let fileManager = FileManager.default
      #if targetEnvironment(macCatalyst)
        guard
          let systemPicturesUrl = fileManager.urls(for: .picturesDirectory, in: .userDomainMask)
            .first
        else { return [] }
      #else
        guard let documentUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return [] }
        let systemPicturesUrl = documentUrl.appendingPathComponent("Pictures")
      #endif
      let directoryUrl =
        directory.isEmpty ? systemPicturesUrl : systemPicturesUrl.appendingPathComponent(directory)
      let fileUrls = try fileManager.contentsOfDirectory(
        at: directoryUrl, includingPropertiesForKeys: nil)
      return fileUrls.map { $0.path }
    }
  }

  func listFilesUnderPictures() -> [String] {
    return forwardExceptionsToJS { () -> [String] in
      let fileManager = FileManager.default
      #if targetEnvironment(macCatalyst)
        guard
          let systemPicturesUrl = fileManager.urls(for: .picturesDirectory, in: .userDomainMask)
            .first
        else { return [] }
      #else
        guard let documentUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return [] }
        let systemPicturesUrl = documentUrl.appendingPathComponent("Pictures")
      #endif
      let fileUrls = try fileManager.contentsOfDirectory(
        at: systemPicturesUrl, includingPropertiesForKeys: nil)
      return fileUrls.map { $0.path }
    }
  }

  func picturesPath() -> String? {
    return forwardExceptionsToJS {
      let fileManager = FileManager.default
      #if targetEnvironment(macCatalyst)
        guard
          let systemPicturesUrl = fileManager.urls(for: .picturesDirectory, in: .userDomainMask)
            .first
        else { return nil }
      #else
        guard let documentUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return nil }
        let systemPicturesUrl = documentUrl.appendingPathComponent("Pictures")
      #endif
      return systemPicturesUrl.path
    }
  }

  func loadImageFileToCanvas(_ file: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      delegate.loadImageFileToCanvas(file)
    }
  }

  func loadImageSrcToCanvas(_ srcContent: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.loadImageSrcToCanvas(srcContent)
    }
  }

  func saveImageFileFromCanvas(_ file: String, _ visibleRegionOnly: Bool) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      delegate.saveImageFileFromCanvas(file, visibleRegionOnly)
    }
  }

  func saveImageSrcFromCanvas(_ visibleRegionOnly: Bool) -> String? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      guard let data = delegate.saveImageDataFromCanvas(visibleRegionOnly) else { return nil }
      return SupportedPrefix.base64Png.rawValue + data.base64EncodedString()
    }
  }

  func existingPrompts() -> [String: String] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let prompts = delegate.existingPrompts()
      return ["prompt": prompts.prompt, "negativePrompt": prompts.negativePrompt]
    }
  }

  func loadLayerFromPhotos(_ type: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.loadLayerFromPhotos(type: type)
    }
  }

  func loadLayerFromFiles(_ type: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.loadLayerFromFiles(type: type)
    }
  }

  func loadLayerFromSrc(_ srcContent: String, _ type: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.loadLayerFromSrc(srcContent, type: type)
    }
  }

  func loadLayerFromJson(_ json: String, _ type: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.loadLayerFromJson(json, type: type)
    }
  }

  func imageSizeFromSrc(_ srcContent: String) -> CGSize {
    guard let delegate = delegate else { return .zero }
    return delegate.imageSizeFromSrc(srcContent)
  }

  func saveLayerSrc(_ type: String) -> String? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      guard let data = try delegate.saveLayerData(type: type) else { return nil }
      return SupportedPrefix.base64Png.rawValue + data.base64EncodedString()
    }
  }

  func extractDepthMap() {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.extractDepthMap()
    }
  }

  func clearMoodboard() {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.clearMoodboard()
    }
  }

  func removeFromMoodboardAt(_ index: Int) {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.removeFromMoodboardAt(index)
    }
  }

  func setMoodboardImageWeight(_ weight: Double, _ index: Int) {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.setMoodboardImageWeight(weight, index)
    }
  }

  func downloadBuiltins(_ files: [String]) {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.downloadBuiltins(files)
    }
  }

  func areModelsDownloaded(_ files: [String]) -> [Bool] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      return try delegate.areModelsDownloaded(files)
    }
  }

  func detectFaces() -> [[String: Any]] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let rects = try delegate.detectFaces().map { JSRect(rect: $0) }
      let data = try JSONEncoder().encode(["faces": rects])
      let jsonObject = try JSONSerialization.jsonObject(with: data) as! [String: Any]
      return jsonObject["faces"] as! [[String: Any]]
    }
  }

  func detectHands() -> [[String: Any]] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let rects = try delegate.detectHands().map { JSRect(rect: $0) }
      let data = try JSONEncoder().encode(["hands": rects])
      let jsonObject = try JSONSerialization.jsonObject(with: data) as! [String: Any]
      return jsonObject["hands"] as! [[String: Any]]
    }
  }

  func requestFromUser(_ title: String, _ confirm: String, _ config: [[String: Any]]) -> [Any] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let (lifetimeObjects, result) = try delegate.requestFromUser(
        title: title, confirm: confirm, config)
      self.lifetimeObjects.append(contentsOf: lifetimeObjects)
      // Do a round-trip copy so it is "clean" JSON object to JS execution engine.
      let data = try JSONSerialization.data(withJSONObject: result)
      let jsonObject = try JSONSerialization.jsonObject(with: data) as! [Any]
      return jsonObject
    }
  }

  func screenSize() -> [String: Any] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      return delegate.screenSize()
    }
  }
}

extension ScriptExecutionSession {
  static func sessionId() -> UInt64 {
    (ScriptExecutor.systemRandomNumberGenerator.next() & 0xffff_ffff_fff0_0000)
      | (UInt64(Date().timeIntervalSince1970) & 0xf_ffff)
  }
}
