import Foundation
import MediaGenerationKit
import ModelZoo
import Upscaler
import NNC
import DataModels
import LocalImageGenerator
import Diffusion
#if canImport(UIKit)
import UIKit
#endif

enum UpscalerError: LocalizedError {
    case modelNotDownloaded
    case imageConversionFailed
    case upscaleFailed
    
    var errorDescription: String? {
        switch self {
        case .modelNotDownloaded: return "Upscaler model is not downloaded."
        case .imageConversionFailed: return "Failed to process image format."
        case .upscaleFailed: return "Upscaling execution failed."
        }
    }
}

@MainActor
final class ImageUpscalerBackend: ObservableObject {
    static let shared = ImageUpscalerBackend()
    
    @Published var isUpscaling = false
    
    private init() {}
    
    func upscale(image: UIImage, model: AIModel) async throws -> UIImage {
        isUpscaling = true
        defer { isUpscaling = false }
        
        guard ModelData.isModelFullyAvailableLocally(model) else {
            throw UpscalerError.modelNotDownloaded
        }
        
        let task = Task.detached(priority: .userInitiated) {
            let upscalerFilePath = ModelZoo.filePathForModelDownloaded(model.id)
            let nativeScaleFactor = UpscalerZoo.scaleFactorForModel(model.id)
            let forcedScaleFactor = nativeScaleFactor
            let numberOfBlocks = UpscalerZoo.numberOfBlocksForModel(model.id)
            
            guard let cgImage = image.cgImage else {
                throw UpscalerError.imageConversionFailed
            }
            
            guard let bitmapContext = ImageConverter.bitmapContext(from: cgImage) else {
                throw UpscalerError.imageConversionFailed
            }
            
            let tensorTuple = ImageConverter.tensor(from: bitmapContext)
            guard let inputTensor = tensorTuple.0 else {
                throw UpscalerError.imageConversionFailed
            }
            
            let graph = DynamicGraph()
            
            let outputTensor = graph.withNoGrad {
                let realESRGANer = RealESRGANer<FloatType>(
                    filePath: upscalerFilePath,
                    nativeScaleFactor: nativeScaleFactor,
                    forcedScaleFactor: forcedScaleFactor,
                    numberOfBlocks: numberOfBlocks,
                    isNHWCPreferred: DeviceCapability.isNHWCPreferred,
                    tileSize: DeviceCapability.RealESRGANerTileSize
                )
                
                let inputVar = graph.variable(inputTensor).toGPU()
                let (outputVar, _) = realESRGANer.upscale(inputVar)
                return outputVar.rawValue.toCPU()
            }
            
            let result = ImageConverter.image(from: outputTensor, scaleFactor: 1.0)
            return result
        }
        
        return try await task.value
    }
}
