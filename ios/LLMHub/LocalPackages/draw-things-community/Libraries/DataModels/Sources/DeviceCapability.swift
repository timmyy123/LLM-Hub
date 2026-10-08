import Atomics
import Diffusion
import Foundation
import Utils

#if canImport(Metal)
  import Metal
#endif

#if canImport(UIKit)
  extension UIDevice {
    var modelIdentifier: String {
      var systemInfo = utsname()
      uname(&systemInfo)
      let machineMirror = Mirror(reflecting: systemInfo.machine)
      let identifier = machineMirror.children.reduce("") { identifier, element in
        guard let value = element.value as? Int8, value != 0 else { return identifier }
        return identifier + String(UnicodeScalar(UInt8(value)))
      }
      return identifier
    }
  }

  import UIKit

  private func isPhone() -> Bool {
    return UIDevice.current.userInterfaceIdiom == .phone
  }
#else
  private func isPhone() -> Bool {
    return false
  }
#endif

private func isMacCatalystBuild() -> Bool {
  #if targetEnvironment(macCatalyst)
    return true
  #else
    return false
  #endif
}

public struct DeviceCapability {
  public enum WeightsStreaming: Int, CaseIterable, Sendable {
    case _8GiB = 8
    case _16GiB = 16
    case _24GiB = 24
    case _32GiB = 32
    case _40GiB = 40
    case _48GiB = 48
    case _56GiB = 56
    case _64GiB = 64
    case _72GiB = 72
    case none = 0
  }

  public static var weightsStreaming: WeightsStreaming {
    let physicalMemory = ProcessInfo.processInfo.physicalMemory
    let gibibyte = UInt64(1_024 * 1_024 * 1_024)
    let reservedMemory = 32 * gibibyte
    let availableMemory = physicalMemory > reservedMemory ? physicalMemory - reservedMemory : 0
    // Allow for the 149.44 GiB V4.1 checkpoint before disabling streaming.
    if availableMemory >= 150 * gibibyte {
      return .none
    } else if availableMemory >= 72 * gibibyte {
      return ._72GiB
    } else if availableMemory >= 64 * gibibyte {
      return ._64GiB
    } else if availableMemory >= 56 * gibibyte {
      return ._56GiB
    } else if availableMemory >= 48 * gibibyte {
      return ._48GiB
    } else if availableMemory >= 40 * gibibyte {
      return ._40GiB
    } else if availableMemory >= 32 * gibibyte {
      return ._32GiB
    } else if availableMemory >= 24 * gibibyte {
      return ._24GiB
    } else if availableMemory >= 16 * gibibyte {
      return ._16GiB
    }
    return ._8GiB
  }

  public static let keepModelPreloaded: Bool = {
    return isHighPerformance
  }()
  public static let isCoreMLSupported: Bool = {
    return false
  }()
  public static let isLoRACoreMLSupported: Bool = {
    return false
  }()
  public static let isMemoryMapBufferSupported: Bool = {
    if #available(iOS 16.2, macOS 13.1, *) {
      return true
    }
    return false
  }()
  public static let isIntel: Bool = {
    #if arch(i386) || arch(x86_64)
      return true
    #else
      return false
    #endif
  }()
  public static let isMFACausalAttentionMaskSupported: Bool = {
    #if arch(i386) || arch(x86_64) || !canImport(Metal)
      return false
    #else
      if #available(iOS 16.5, macOS 13.4, macCatalyst 16.5, *) {
        return true
      }
      return false
    #endif
  }()
  public static let isNHWCPreferred: Bool = {
    #if canImport(Metal)
      return true
    #else
      return false
    #endif
  }()
  // Only enable this for more than 12GiB RAM and 18 above.
  public static let isTAESDPreferred: Bool = {
    #if arch(i386) || arch(x86_64)
      return false
    #else
      if #available(iOS 18, macOS 15, macCatalyst 18, *) {
        // Gate to M2 and above.
        if isMaxPerformance, let device = MTLCreateSystemDefaultDevice(),
          device.supportsFamily(.apple8)
        {
          return true
        }
        return false
      } else {
        return false
      }
    #endif
  }()
  public static let isInteractiveGenerationSupported: Bool = {
    #if arch(i386) || arch(x86_64) || !canImport(Metal)
      return false
    #else
      if #available(iOS 26, macOS 26, macCatalyst 26, *) {
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        guard physicalMemory >= 50_465_865_728 else {
          return false
        }
        if let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple10) {
          return true
        }
      }
      return false
    #endif
  }()
  public static let isMFASupported: Bool = {
    #if !canImport(Metal)
      return true
    #elseif arch(i386) || arch(x86_64)
      return false
    #else
      if #available(iOS 16, macOS 13, macCatalyst 16, *) {
        if let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7) {
          return true
        }
        return false
      }
      return false
    #endif
  }()
  public static let isMQASupported: Bool = {
    #if !canImport(Metal)
      return false
    #elseif arch(i386) || arch(x86_64)
      return false
    #else
      if #available(iOS 26.2, macOS 26.2, macCatalyst 26.2, *) {
        if let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple10) {
          return true
        }
        return false
      }
      return false
    #endif
  }()
  public static let isM3OrLater: Bool = {
    #if arch(i386) || arch(x86_64) || !canImport(Metal)
      return false
    #else
      if #available(iOS 16, macOS 13, macCatalyst 16, *) {
        if let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple9) {
          return true
        }
        return false
      }
      return false
    #endif
  }()
  public static let isMFAGEMMFaster: Bool = {
    #if arch(i386) || arch(x86_64) || !canImport(Metal)
      return false
    #else
      if #available(iOS 16, macOS 13, macCatalyst 16, *) {
        if let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7) {
          return true
        }
        return false
      }
      return false
    #endif
  }()
  public static let isMFAAttentionFaster: Bool = {
    #if arch(i386) || arch(x86_64) || !canImport(Metal)
      return false
    #else
      if #available(iOS 16, macOS 13, macCatalyst 16, *) {
        if let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7) {
          return true
        }
        return false
      }
      return false
    #endif
  }()
  public static let isMFAAppleNeuralEngineFaster: Bool = {
    #if arch(i386) || arch(x86_64) || !canImport(Metal)
      return false
    #else
      if #available(iOS 26, macOS 26, macCatalyst 26, *) {
        if let device = MTLCreateSystemDefaultDevice(),
          device.supportsFamily(.apple9) && !device.supportsFamily(.apple10)
        {
          return true
        }
        return false
      }
      return false
    #endif
  }()
  public static let isMFAEnabled = ManagedAtomic(isMFASupported ? 1 : 0)
  public static let isMFAAppleNeuralEngineEnabled = ManagedAtomic(isMFAAppleNeuralEngineFaster)
  public struct Scale: Equatable & Hashable & CustomDebugStringConvertible {
    public let widthScale: UInt16
    public let heightScale: UInt16
    public var debugDescription: String {
      let gcd = UInt16.gcd(heightScale, widthScale)
      return "\(widthScale / gcd):\(heightScale / gcd)"
    }
    public init(widthScale: UInt16, heightScale: UInt16) {
      self.widthScale = widthScale
      self.heightScale = heightScale
    }
    public func aspectCompatibleScale(for defaultScale: UInt16) -> Scale {
      guard defaultScale > max(widthScale, heightScale) else { return self }
      if widthScale >= heightScale {
        return Scale(
          widthScale: defaultScale,
          heightScale: UInt16(
            (Double(defaultScale) / Double(widthScale) * Double(heightScale)).rounded(.up)))
      } else {
        return Scale(
          widthScale: UInt16(
            (Double(defaultScale) / Double(heightScale) * Double(widthScale)).rounded(.up)),
          heightScale: defaultScale)
      }
    }
    public func area() -> UInt16 {
      return widthScale * heightScale
    }
  }
  public static let `default`: Scale = {
    if isGoodPerformance {
      return Scale(widthScale: 8, heightScale: 8)
    }
    // Should be 4GiB devices and below.
    return Scale(widthScale: 6, heightScale: 6)
  }()
  public static let isLowPerformance: Bool = {
    #if !canImport(Metal)
      return false
    #else
      let physicalMemory = ProcessInfo.processInfo.physicalMemory
      // Should be 3GiB devices.
      return physicalMemory < 3_758_096_384  // This is 3.5 * 1024 * 1024 * 1024.
    #endif
  }()
  public static let isGoodPerformance: Bool = {
    #if !canImport(Metal)
      return true
    #else
      let physicalMemory = ProcessInfo.processInfo.physicalMemory
      // Should be 6GiB and above devices.
      return physicalMemory >= 5_368_709_120  // This is 5 * 1024 * 1024 * 1024.
    #endif
  }()
  public static let isHighPerformance: Bool = {
    #if !canImport(Metal)
      return true
    #else
      let physicalMemory = ProcessInfo.processInfo.physicalMemory
      return physicalMemory >= 7_516_192_768  // This is 7 * 1024 * 1024 * 1024.
    #endif
  }()
  public static let isMaxPerformance: Bool = {
    #if !canImport(Metal)
      return true
    #else
      let physicalMemory = ProcessInfo.processInfo.physicalMemory
      return physicalMemory >= 11_811_160_064  // This is 11 * 1024 * 1024 * 1024.
    #endif
  }()
  public static let isUltraPerformance: Bool = {
    #if !canImport(Metal)
      return true
    #else
      let physicalMemory = ProcessInfo.processInfo.physicalMemory
      return physicalMemory >= 24_696_061_952  // This is 23 * 1024 * 1024 * 1024.
    #endif
  }()
  public static let shouldConserveMemoryForSmallLLM: Bool = {  // Small here means 20~30B.
    #if arch(i386) || arch(x86_64) || !canImport(Metal)
      return true
    #else
      if #available(iOS 26, macOS 26, macCatalyst 26, *) {
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        return physicalMemory < 50_465_865_728
      }
      return true
    #endif
  }()
  public static let isPrefillWarmupPreferred: Bool = {
    let physicalMemory = ProcessInfo.processInfo.physicalMemory
    return physicalMemory >= 128 * 1_024 * 1_024 * 1_024
  }()
  public static var memoryCapacity: MemoryCapacity = {
    let physicalMemory = ProcessInfo.processInfo.physicalMemory
    if physicalMemory >= 33_285_996_544 {  // This is 31 * 1024 * 1024 * 1024.
      return .veryHigh
    } else if physicalMemory >= 24_696_061_952 {  // This is 23 * 1024 * 1024 * 1024.
      return .high
    } else if physicalMemory >= 16_106_127_360 {  // This is 15 * 1024 * 1024 * 1024.
      return .medium
    }
    return .low
  }()
  public static var isFreadPreferred: Bool = isUMA
  public static var cacheUri: URL = {
    let urls = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
    return urls.first ?? URL(fileURLWithPath: NSTemporaryDirectory())
  }()
  public static let isPartialOffloadPreferred = ManagedAtomic(false)
  public static var deviceProperties: DeviceProperties {
    DeviceProperties(
      isFreadPreferred: isFreadPreferred, memoryCapacity: memoryCapacity,
      isNHWCPreferred: isNHWCPreferred, cacheUri: cacheUri,
      isPartialOffloadPreferred: isPartialOffloadPreferred.load(ordering: .acquiring))
  }
  public static var maxTotalWeightsCacheSize: UInt64 = {
    #if !canImport(Metal)
      return 0
    #else
      let physicalMemory = ProcessInfo.processInfo.physicalMemory
      // This is 47 * 1024 * 1024 * 1024
      guard physicalMemory >= 50_465_865_728 else {
        return 0
      }
      // Make it half and rounded it to multiple of 8.
      let residualMemory = physicalMemory / 1_024 / 1_024 / 1_024 / 2 / 8
      return residualMemory * 8_589_934_592
    #endif
  }()
  public static let isUMA: Bool = {
    #if canImport(Metal)
      #if arch(i386) || arch(x86_64)
        return false
      #else
        return true
      #endif
    #else
      return false
    #endif
  }()
  public static let RealESRGANerTileSize: Int = {  // Metal have problem to upscale to 2048x2048, hence doing this for 1024x1024 on Metal. CUDA doesn't have this issue and will benefit from larger tiles.
    #if canImport(Metal)
      return 256
    #else
      return 512
    #endif
  }()
  public static func isVerifiedScale(_ scale: Scale) -> Bool {
    // Only for low performance devices, we need to gate against anything above 512x512.
    if !isLowPerformance || scale.widthScale * scale.heightScale <= 64 {
      return true
    }
    return false
  }
  public static func defaultScale(_ defaultScale: UInt16) -> UInt16 {
    switch defaultScale {
    case 16:
      return 16
    case 12:
      return 12
    case 8:
      return 8
    default:
      return defaultScale
    }
  }
  public static var scales: [Scale] = {
    var scales = [Scale]()
    scales.append(Scale(widthScale: 6, heightScale: 4))
    scales.append(Scale(widthScale: 4, heightScale: 4))
    scales.append(Scale(widthScale: 4, heightScale: 6))
    scales.append(Scale(widthScale: 8, heightScale: 4))
    scales.append(Scale(widthScale: 4, heightScale: 8))
    scales.append(Scale(widthScale: 8, heightScale: 5))
    scales.append(Scale(widthScale: 6, heightScale: 6))
    scales.append(Scale(widthScale: 5, heightScale: 8))
    scales.append(Scale(widthScale: 7, heightScale: 7))
    scales.append(Scale(widthScale: 10, heightScale: 5))
    scales.append(Scale(widthScale: 5, heightScale: 10))
    scales.append(Scale(widthScale: 9, heightScale: 6))
    scales.append(Scale(widthScale: 6, heightScale: 9))
    if !isLowPerformance {
      scales.append(Scale(widthScale: 12, heightScale: 8))
    }
    scales.append(Scale(widthScale: 8, heightScale: 8))
    if !isLowPerformance {
      scales.append(Scale(widthScale: 8, heightScale: 12))
    }
    scales.append(Scale(widthScale: 12, heightScale: 6))
    scales.append(Scale(widthScale: 6, heightScale: 12))
    guard !isLowPerformance else {
      return scales
    }
    scales.append(Scale(widthScale: 10, heightScale: 10))
    scales.append(Scale(widthScale: 15, heightScale: 10))
    scales.append(Scale(widthScale: 10, heightScale: 15))
    scales.append(Scale(widthScale: 12, heightScale: 12))
    scales.append(Scale(widthScale: 16, heightScale: 8))
    scales.append(Scale(widthScale: 8, heightScale: 16))
    scales.append(Scale(widthScale: 16, heightScale: 9))
    scales.append(Scale(widthScale: 9, heightScale: 16))
    guard isGoodPerformance else {
      return scales
    }
    scales.append(Scale(widthScale: 16, heightScale: 10))
    scales.append(Scale(widthScale: 10, heightScale: 16))
    scales.append(Scale(widthScale: 18, heightScale: 10))
    scales.append(Scale(widthScale: 10, heightScale: 18))
    scales.append(Scale(widthScale: 18, heightScale: 12))
    scales.append(Scale(widthScale: 15, heightScale: 15))
    scales.append(Scale(widthScale: 12, heightScale: 18))
    scales.append(Scale(widthScale: 20, heightScale: 10))
    scales.append(Scale(widthScale: 10, heightScale: 20))
    scales.append(Scale(widthScale: 22, heightScale: 11))
    scales.append(Scale(widthScale: 21, heightScale: 12))
    scales.append(Scale(widthScale: 19, heightScale: 13))
    scales.append(Scale(widthScale: 18, heightScale: 14))
    scales.append(Scale(widthScale: 16, heightScale: 16))
    scales.append(Scale(widthScale: 14, heightScale: 18))
    scales.append(Scale(widthScale: 13, heightScale: 19))
    scales.append(Scale(widthScale: 12, heightScale: 21))
    scales.append(Scale(widthScale: 11, heightScale: 22))
    return scales
  }()

  public static func maxSupportScale() -> UInt16 {
    var maxScale: UInt16 = 10
    if let maxWidthScale = Self.scales.max(by: { a, b in a.widthScale < b.widthScale }) {
      maxScale = max(maxScale, maxWidthScale.widthScale)
    }

    if let maxHeightScale = Self.scales.max(by: { a, b in a.heightScale < b.heightScale }) {
      maxScale = max(maxScale, maxHeightScale.heightScale)
    }
    return maxScale
  }

  public static var availableDecodingTileScales: [Scale] {
    var scales = [Scale]()
    scales.append(Scale(widthScale: 6, heightScale: 6))
    if !DeviceCapability.isLowPerformance {
      scales.append(Scale(widthScale: 8, heightScale: 8))
    }
    if DeviceCapability.isGoodPerformance {
      scales.append(Scale(widthScale: 10, heightScale: 10))
    }
    if DeviceCapability.isHighPerformance {
      scales.append(Scale(widthScale: 12, heightScale: 12))
      scales.append(Scale(widthScale: 15, heightScale: 15))
    }
    if DeviceCapability.isMaxPerformance {
      scales.append(Scale(widthScale: 16, heightScale: 16))
      scales.append(Scale(widthScale: 20, heightScale: 20))
    }
    return scales
  }

  public static var availableDiffusionTileScales: [Scale] {
    var scales = [Scale]()
    scales.append(Scale(widthScale: 6, heightScale: 6))
    scales.append(Scale(widthScale: 8, heightScale: 8))
    if !DeviceCapability.isLowPerformance {
      scales.append(Scale(widthScale: 10, heightScale: 10))
    }
    if DeviceCapability.isGoodPerformance {
      scales.append(Scale(widthScale: 12, heightScale: 12))
    }
    if DeviceCapability.isHighPerformance {
      scales.append(Scale(widthScale: 16, heightScale: 16))
    }
    if DeviceCapability.isMaxPerformance {
      scales.append(Scale(widthScale: 20, heightScale: 20))
    }
    return scales
  }

  public static func firstPassHighResolutionFixScales(for scale: Scale) -> [Scale] {
    if scale.widthScale == scale.heightScale {
      var scales = [Scale]()
      if scale.widthScale > 16 {
        scales.append(Scale(widthScale: 16, heightScale: 16))
      }
      if scale.widthScale > 15 {
        scales.append(Scale(widthScale: 15, heightScale: 15))
      }
      if scale.widthScale > 12 {
        scales.append(Scale(widthScale: 12, heightScale: 12))
      }
      if scale.widthScale > 10 {
        scales.append(Scale(widthScale: 10, heightScale: 10))
      }
      if scale.widthScale > 8 {
        scales.append(Scale(widthScale: 8, heightScale: 8))
      }
      if scale.widthScale > 7 {
        scales.append(Scale(widthScale: 7, heightScale: 7))
      }
      if scale.widthScale > 6 {
        scales.append(Scale(widthScale: 6, heightScale: 6))
      }
      if scale.widthScale > 4 {
        scales.append(Scale(widthScale: 4, heightScale: 4))
      }
      return scales
    } else if scale.widthScale * 2 == scale.heightScale * 3 {
      var scales = [Scale]()
      if scale.widthScale > 18 {
        scales.append(Scale(widthScale: 18, heightScale: 12))
      }
      if scale.widthScale > 15 {
        scales.append(Scale(widthScale: 15, heightScale: 10))
      }
      if scale.widthScale > 12 {
        scales.append(Scale(widthScale: 12, heightScale: 8))
      }
      if scale.widthScale > 9 {
        scales.append(Scale(widthScale: 9, heightScale: 6))
      }
      if scale.widthScale > 6 {
        scales.append(Scale(widthScale: 6, heightScale: 4))
      }
      return scales
    } else if scale.widthScale * 3 == scale.heightScale * 2 {
      var scales = [Scale]()
      if scale.heightScale > 18 {
        scales.append(Scale(widthScale: 12, heightScale: 18))
      }
      if scale.heightScale > 15 {
        scales.append(Scale(widthScale: 10, heightScale: 15))
      }
      if scale.heightScale > 12 {
        scales.append(Scale(widthScale: 8, heightScale: 12))
      }
      if scale.heightScale > 9 {
        scales.append(Scale(widthScale: 6, heightScale: 9))
      }
      if scale.heightScale > 6 {
        scales.append(Scale(widthScale: 4, heightScale: 6))
      }
      return scales
    } else if scale.widthScale * 3 == scale.heightScale * 4 {
      var scales = [Scale]()
      if scale.widthScale > 16 {
        scales.append(Scale(widthScale: 16, heightScale: 12))
      }
      if scale.widthScale > 12 {
        scales.append(Scale(widthScale: 12, heightScale: 9))
      }
      if scale.widthScale > 8 {
        scales.append(Scale(widthScale: 8, heightScale: 6))
      }
      return scales
    } else if scale.widthScale * 4 == scale.heightScale * 3 {
      var scales = [Scale]()
      if scale.heightScale > 16 {
        scales.append(Scale(widthScale: 12, heightScale: 16))
      }
      if scale.heightScale > 12 {
        scales.append(Scale(widthScale: 9, heightScale: 12))
      }
      if scale.heightScale > 8 {
        scales.append(Scale(widthScale: 6, heightScale: 8))
      }
      return scales
    } else if scale.widthScale * 5 == scale.heightScale * 8 {
      if scale.widthScale > 8 {
        return [Scale(widthScale: 8, heightScale: 5)]
      }
      return []
    } else if scale.widthScale * 8 == scale.heightScale * 5 {
      if scale.heightScale > 8 {
        return [Scale(widthScale: 5, heightScale: 8)]
      }
      return []
    } else if scale.widthScale == scale.heightScale * 2 {
      var scales = [Scale]()
      if scale.heightScale > 14 {
        scales.append(Scale(widthScale: 28, heightScale: 14))
      }
      if scale.heightScale > 11 {
        scales.append(Scale(widthScale: 22, heightScale: 11))
      }
      if scale.heightScale > 8 {
        scales.append(Scale(widthScale: 16, heightScale: 8))
      }
      if scale.heightScale > 6 {
        scales.append(Scale(widthScale: 12, heightScale: 6))
      }
      if scale.heightScale > 5 {
        scales.append(Scale(widthScale: 10, heightScale: 5))
      }
      if scale.heightScale > 4 {
        scales.append(Scale(widthScale: 8, heightScale: 4))
      }
      return scales
    } else if scale.widthScale * 2 == scale.heightScale {
      var scales = [Scale]()
      if scale.widthScale > 14 {
        scales.append(Scale(widthScale: 14, heightScale: 28))
      }
      if scale.widthScale > 11 {
        scales.append(Scale(widthScale: 11, heightScale: 22))
      }
      if scale.widthScale > 8 {
        scales.append(Scale(widthScale: 8, heightScale: 16))
      }
      if scale.widthScale > 6 {
        scales.append(Scale(widthScale: 6, heightScale: 12))
      }
      if scale.widthScale > 5 {
        scales.append(Scale(widthScale: 5, heightScale: 10))
      }
      if scale.widthScale > 4 {
        scales.append(Scale(widthScale: 4, heightScale: 8))
      }
      return scales
    }
    return []
  }

  public static func isHighPrecisionVAEFallbackEnabled(scale: Scale) -> Bool {
    if isMaxPerformance {
      return true
    }
    if isHighPerformance && scale.widthScale * scale.heightScale <= 256 {
      return true
    }
    if isGoodPerformance && scale.widthScale * scale.heightScale <= 100 {
      return true
    }
    return false
  }

  public static func externalOnDemand(
    version: ModelVersion, scale: Scale, force: Bool, suffix: String? = nil,
    is8BitModel: Bool = false
  )
    -> Bool
  {
    switch version {
    case .auraflow:  // AuraFlow is a big model, we may need different logic for this.
      guard (!isHighPerformance && !(isGoodPerformance && is8BitModel)) || force else {
        return false
      }
    case .flux1:
      guard (!isMaxPerformance && !(isHighPerformance && is8BitModel)) || force
      else {
        return false
      }
    case .hunyuanVideo:
      guard
        (!isMaxPerformance
          && !((isMacCatalystBuild() ? isHighPerformance : isMaxPerformance) && is8BitModel))
          || force
      else {
        return false
      }
    case .sd3Large:
      guard (!isMaxPerformance && !(isHighPerformance && is8BitModel)) || force
      else {
        return false
      }
    case .sdxlBase, .sd3, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB:
      guard (!isHighPerformance && !(isGoodPerformance && is8BitModel)) || force else {
        return false
      }
    case .v1, .v2, .pixart, .kandinsky21:
      guard (!isGoodPerformance && !is8BitModel) || force else {
        return false
      }
      // If it is not low performance but also not good performance, we check if it is not v2 model
      // or Kandinsky and the size is larger than 5x8, we use file-backed.
      if !force && !isLowPerformance && version == .v1 && scale.widthScale * scale.heightScale <= 40
      {
        return false
      }
    case .wan21_1_3b:
      guard (!isHighPerformance && !(isGoodPerformance && is8BitModel)) || force else {
        return false
      }
    case .wan22_5b:
      guard (!isUltraPerformance && !(isMaxPerformance && is8BitModel)) || force else {
        return false
      }
    case .zImage, .qwenImage2_1, .ernieImage, .ideogram4, .krea2:
      guard (!isUltraPerformance && !(isMaxPerformance && is8BitModel)) || force else {
        return false
      }
    case .wan21_14b, .longcatVideoAvatar1_5:
      guard
        (!isUltraPerformance && !(isMaxPerformance && is8BitModel)) || force
      else {
        return false
      }
    case .qwenImage, .hiDreamO1:
      guard
        (!isUltraPerformance && !(isMaxPerformance && is8BitModel)) || force
      else {
        return false
      }
    case .hiDreamI1:
      guard
        (!isUltraPerformance && !(isHighPerformance && is8BitModel)) || force
      else {
        return false
      }
    case .flux2, .flux2_9b:
      guard
        (!isUltraPerformance && !(isMaxPerformance && is8BitModel)) || force
      else {
        return false
      }
    case .flux2_4b, .cosmos2_5_2b, .seedvr2_3b, .seedvr2_7b:
      guard
        (!isMaxPerformance && !(isHighPerformance && is8BitModel)) || force
      else {
        return false
      }
    case .ltx2, .ltx2_3:
      guard !isUltraPerformance || force else {
        return false
      }
    case .minimaxH3:
      guard !isUltraPerformance || force else {
        return false
      }
    }
    return true
  }
}
