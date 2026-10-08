import Foundation

public func jsonRepresentation(configuration: GenerationConfiguration) -> String {
  let dictionary: [String: String] = [
    "startWidth": "\(configuration.startWidth)",
    "startHeight": "\(configuration.startHeight)",
    "seed": "\(configuration.seed)",
    "steps": "\(configuration.steps)",
    "guidanceScale": "\(configuration.guidanceScale)",
    "strength": "\(configuration.strength)",
    "model": "\(configuration.model ?? "none")",
    "sampler": "\(configuration.sampler)",
    "batchCount": "\(configuration.batchCount)",
    "batchSize": "\(configuration.batchSize)",
    "hiresFix": "\(configuration.hiresFix)",
    "hiresFixStartWidth": "\(configuration.hiresFixStartWidth)",
    "hiresFixStartHeight": "\(configuration.hiresFixStartHeight)",
    "hiresFixStrength": "\(configuration.hiresFixStrength)",
    "upscaler": "\(configuration.upscaler ?? "none")",
    "imageGuidanceScale": "\(configuration.imageGuidanceScale)",
    "seedMode": "\(configuration.seedMode)",
    "clipSkip": "\(configuration.clipSkip)",
    "controls": "\(configuration.controls.map { $0.file ?? "none" })",
    "loras": "\(configuration.loras.map { $0.file ?? "none" })",
    "maskBlur": "\(configuration.maskBlur)",
    "maskBlurOutset": "\(configuration.maskBlurOutset)",
    "sharpness": "\(configuration.sharpness)",
    "faceRestoration": "\(configuration.faceRestoration ?? "none")",
    "clipWeight": "\(configuration.clipWeight)",
    "negativePromptForImagePrior": "\(configuration.negativePromptForImagePrior)",
    "imagePriorSteps": "\(configuration.imagePriorSteps)",
  ]
  let data = try! JSONSerialization.data(withJSONObject: dictionary)
  return String(data: data, encoding: .utf8)!
}
