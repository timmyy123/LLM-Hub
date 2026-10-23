import Foundation
import Metal

enum ModelFit { case gpu, partial, tooLarge }

struct HardwareInfo {
    let ramBytes: UInt64
    /// On Apple Silicon this is the share of unified memory Metal lets the GPU use.
    let gpuBytes: UInt64
    let gpuName: String
    let isAppleSilicon: Bool

    static func detect() -> HardwareInfo {
        let device = MTLCreateSystemDefaultDevice()
        var arm = false
        #if arch(arm64)
        arm = true
        #endif
        let info = HardwareInfo(
            ramBytes: ProcessInfo.processInfo.physicalMemory,
            gpuBytes: device?.recommendedMaxWorkingSetSize ?? 0,
            gpuName: device?.name ?? "",
            isAppleSilicon: arm)
        Log.info("Hardware: RAM \(info.ramGB) GB, GPU '\(info.gpuName)' working set \(info.gpuGB) GB, arm64=\(arm)")
        return info
    }

    var ramGB: Int { Int((Double(ramBytes) / 1_073_741_824).rounded()) }
    var gpuGB: Int { Int((Double(gpuBytes) / 1_073_741_824).rounded()) }
    var hasDedicatedGpu: Bool { gpuBytes >= 4 << 30 }

    func fit(_ modelBytes: Int64) -> ModelFit {
        // Weights plus room for the KV cache and runtime buffers.
        let needed = Double(modelBytes) * 1.2
        if gpuBytes > 0 && needed <= Double(gpuBytes) * 0.92 { return .gpu }
        if needed <= Double(gpuBytes) * 0.92 + Double(ramBytes) * 0.25 { return .partial }
        return .tooLarge
    }

    /// Only offer quants that run fully on the GPU.
    func shouldOffer(_ fit: ModelFit) -> Bool { hasDedicatedGpu ? fit == .gpu : fit != .tooLarge }
}
