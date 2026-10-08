import CoreGraphics
import Diffusion
import Foundation
import ImageIO
import Invocation
import LocalImageGenerator
import NNC

func decodeParameters(
  fromContainer container: KeyedDecodingContainer<JSONKey>, parameters: Parameters
) throws -> [JSONKey] {
  var keysDecoded: [JSONKey] = []
  for parameter in parameters.allParameters() {
    let jsonKey = parameter.commandLineFlag.replacingOccurrences(of: "-", with: "_")
    let keyStrings: [String] = [jsonKey] + parameter.additionalJsonKeys
    let presentKeys = keyStrings.map { JSONKey($0) }.filter { container.contains($0) }
    guard let firstKey = presentKeys.first else {
      continue
    }
    if presentKeys.count > 1 {
      throw
        "More than one key for \(parameter.title) specified (must only specify one of \(keyStrings))"
    } else {
      try parameter.decode(from: container, forKey: firstKey)
      try parameter.validate()
      keysDecoded.append(firstKey)
    }
  }
  return keysDecoded
}

// TODO: we use the same body here for both txt2img and img2img, even though in sd-webui they take slightly different parameters.
// Ones in txt2img but not img2img: ["enable_hr", "firstphase_width", "firstphase_height", "hr_scale", "hr_upscaler", "hr_second_pass_steps", "hr_resize_x", "hr_resize_y", "hr_sampler_name", "hr_prompt", "hr_negative_prompt"]
// Ones in img2img but not txt2img: ["init_images", "resize_mode", "image_cfg_scale", "mask", "mask_blur", "inpainting_fill", "inpaint_full_res", "inpaint_full_res_padding", "inpainting_mask_invert", "initial_noise_multiplier", "include_init_images"]
final class RequestBody: Decodable {
  private static var parameters: Parameters!

  let images: [Tensor<FloatType>]?
  let restoreFaces: Bool

  static func decodeInitImages(container: KeyedDecodingContainer<JSONKey>) throws -> [Tensor<
    FloatType
  >] {
    let initImages: [Data] = try container.decode([Data].self, forKey: JSONKey("init_images"))
    return try initImages.map { imageData in
      guard let imageSource = CGImageSourceCreateWithData(imageData as CFData, nil),
        let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil as CFDictionary?)
      else {
        throw "Data in init_images did not represent a valid image"
      }
      guard let bitmapContext = ImageConverter.bitmapContext(from: cgImage),
        let image = ImageConverter.tensor(from: bitmapContext).0
      else {
        throw "Failed to convert image into tensor"
      }
      return image
    }
  }

  static func createRequestBody(data: Data, parameters: Parameters) throws -> RequestBody {
    Self.parameters = parameters
    return try JSONDecoder().decode(RequestBody.self, from: data)
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: JSONKey.self)

    let otherKeys = ["restore_faces", "init_images"].map { JSONKey($0) }
    restoreFaces =
      try container.decodeIfPresent(Bool.self, forKey: JSONKey("restore_faces")) ?? false
    images =
      container.contains(JSONKey("init_images"))
      ? try Self.decodeInitImages(container: container) : nil
    let keysDecoded: [JSONKey] =
      try decodeParameters(fromContainer: container, parameters: Self.parameters) + otherKeys
    let keysRemaining: [JSONKey] = container.allKeys.filter { key in
      return !keysDecoded.contains { $0.stringValue == key.stringValue }
    }
    if keysRemaining.count > 0 {
      throw "Unrecognized keys: \(keysRemaining.map(\.stringValue))"
    }
  }
}

final class SuccessResponse: Encodable {
  let images: [Data]

  init(images: [Data]) {
    self.images = images
  }
}

// The error response of sd-webui is not documented as what we have here, but in practice it seems to match this
final class ErrorResponse: Encodable {
  let error: String
  let detail: String
  let body: String
  let errors: String

  init(error: String, detail: String, body: String, errors: String) {
    self.error = error
    self.detail = detail
    self.body = body
    self.errors = errors
  }
}
