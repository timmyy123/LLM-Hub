import Diffusion
import Foundation
import NNC

public struct TextualInversionZoo: DownloadZoo {
  public struct Specification: Codable {
    public var name: String
    public var file: String
    public var keyword: String
    public var length: Int
    public var version: ModelVersion
    public var deprecated: Bool?
    public init(
      name: String, file: String, keyword: String, length: Int, version: ModelVersion,
      deprecated: Bool? = false
    ) {
      self.name = name
      self.file = file
      self.keyword = keyword
      self.length = length
      self.version = version
      self.deprecated = deprecated
    }
  }

  private static var fileSHA256: [String: String] = [
    "carhelper_ti_f16.ckpt":
      "1d9c8434a3670f3a9deff463b59286514a04746f163946c43de5801f968664c5",
    "charturner_ti_f16.ckpt":
      "9155904e44c0329ebdea8885b9729bf0e5dae5c627e5acfcb13f31ad86383e0f",
    "cloudport_v1.0_ti_f16.ckpt":
      "a38337995db9714ca1d9d7d94022fedb3db0c9168a03601ce0aa7f94cdbccf49",
    "double_exposure_ti_f16.ckpt":
      "e0ae6626d444e420f55d2b5e111ac341ccb3bd1c825d1e98748f532dcfdc1b9a",
    "drd_point_e_768_v_ti_f16.ckpt":
      "508b1e299fe57facedba37f039b7fe69e2f2fd1f6d00135399b7b691ae4ce6f6",
    "pure_eros_ti_f16.ckpt":
      "71514c3f85e99bee61f7c2e6218c14bcef5abaa010f11dd6bf290454ff1fbc72",
    "sd2_papercut_ti_f16.ckpt":
      "63435a02c05306183e24909caa5cd48678b5d649c9f7d4c57b50c969c94e27cf",
    "birb_style_ti_f16.ckpt":
      "a2029299f377aef2dda1bb8e994d868af13b025d727676a2a3bc7417fbd5ea85",
    "knollingcase_v4_kc16_5000_ti_f16.ckpt":
      "c8041e652941a0874d178f26fb6fd6ef60b5215b336fad5b54ad2e7543f24769",
    "laxpeint_v2_ti_f16.ckpt":
      "36cb7e2d23238dba0f24607522b3a286aa32469c2083b0da016d332573e52e23",
    "classipeint_ti_f16.ckpt":
      "e1eb0ac8dbd82c5734645164071a4812ace48d03c1ca345410e502e639b2b250",
    "parchart_ti_f16.ckpt":
      "dc50bd7af6e24f8278c74ff9619404aa2da8fb60a64417731499585725460888",
    "cinemahelper_ti_f16.ckpt":
      "e7be077a65aed7763b1d9a2207e8c6b60a4727bf43ba2cca3a8e966096cda390",
    "animescreencap_ti_f16.ckpt":
      "e1355bbec52b755240ec7c860683d843ed8c0021f4ca3e6d2bdf25c044c021ff",
    "v2_dreamink_ti_f16.ckpt":
      "af9319aebe884dbc2e8a82afd40c8bc87fa838e0167aa76f2f3d2b61f5db7aa9",
    "photohelper_ti_f16.ckpt":
      "d928695339f094a8f685d9b4e01366ea05589701533ad6e9cc5c9d193c6089a3",
    "vintagehelper_ti_f16.ckpt":
      "3c4e3c989cac24c05d677b8ca7bd04738d7f00e0d53210d16b398583ed7db8b6",
    "actionhelper_ti_f16.ckpt":
      "bccaf7ed6a5c2f35a4e80c62567eb0b221f6e68fdb066630446b364d14fbc550",
    "bad_prompt_v2_ti_f16.ckpt":
      "4eedda1636bd07f4f5e4da1451fba82476420c944203122501f667c54ff6fe8e",
  ]

  public static let builtinSpecifications: [Specification] = []

  private static let builtinModelsAndAvailableSpecifications: (Set<String>, [Specification]) = {
    let jsonFile = filePathForModelDownloaded("custom_textual_inversions.json")
    guard let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) else {
      return (Set(builtinSpecifications.map { $0.file }), builtinSpecifications)
    }

    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let jsonSpecifications = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value })
    else {
      return (Set(builtinSpecifications.map { $0.file }), builtinSpecifications)
    }

    var availableSpecifications = builtinSpecifications
    var builtinModels = Set(builtinSpecifications.map { $0.file })
    for specification in jsonSpecifications {
      if builtinModels.contains(specification.file) {
        availableSpecifications = availableSpecifications.filter { $0.file != specification.file }
      }
      availableSpecifications.append(specification)
    }
    return (builtinModels, availableSpecifications)
  }()

  private static let builtinModels: Set<String> = builtinModelsAndAvailableSpecifications.0
  public static var availableSpecifications: [Specification] =
    builtinModelsAndAvailableSpecifications.1

  public static func isBuiltinTextualInversion(_ name: String) -> Bool {
    return builtinModels.contains(name)
  }

  public static func mergeFileSHA256(_ sha256: [String: String]) {
    var fileSHA256 = fileSHA256
    for (key, value) in sha256 {
      fileSHA256[key] = value
    }
    self.fileSHA256 = fileSHA256
  }

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in availableSpecifications {
      mapping[specification.file] = specification
    }
    return mapping
  }()

  private static var fileMappingFromKeyword: [String: [String]] = {
    var mapping = [String: [String]]()
    for specification in availableSpecifications {
      var files = mapping[specification.keyword, default: []]
      files.append(specification.file)
      mapping[specification.keyword] = files
    }
    return mapping
  }()

  public static func isModelDeprecated(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.deprecated ?? false
  }

  public static func appendCustomSpecification(_ specification: Specification) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = TextualInversionZoo.filePathForModelDownloaded("custom_textual_inversions.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      if let jsonSpecification = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value }) {
        customSpecifications.append(contentsOf: jsonSpecification)
      }
    }
    customSpecifications = customSpecifications.filter { $0.file != specification.file }
    customSpecifications.append(specification)
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    guard let jsonData = try? jsonEncoder.encode(customSpecifications) else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)
    // Modify these two are not thread safe. availableSpecifications are OK. specificationMapping is particularly problematic (as it is access on both main thread and a background thread).
    var availableSpecifications = availableSpecifications
    availableSpecifications = availableSpecifications.filter { $0.file != specification.file }
    availableSpecifications.append(specification)
    self.availableSpecifications = availableSpecifications
    specificationMapping[specification.file] = specification
    var files = fileMappingFromKeyword[specification.keyword, default: []]
    files.append(specification.file)
    fileMappingFromKeyword[specification.keyword] = files
  }

  public static func filePathForModelDownloaded(_ name: String) -> String {
    return ModelZoo.filePathForModelDownloaded(name)
  }

  public static func isModelDownloaded(_ name: String, memorizedBy: Set<String>) -> Bool {
    return ModelZoo.isModelDownloaded(name, memorizedBy: memorizedBy)
  }

  public static func modelFromKeyword(_ keyword: String, potentials: [String]) -> String? {
    guard !potentials.isEmpty else {
      return fileMappingFromKeyword[keyword]?.last
    }
    guard let files = fileMappingFromKeyword[keyword] else { return nil }
    let existingFiles = Set(files)
    for potential in potentials {
      if existingFiles.contains(potential) {
        return potential
      }
    }
    return files.last
  }

  public static func humanReadableNameForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return name }
    return specification.name
  }

  // We prefer these if it is a hit.
  public static var overrideMapping: [String: Specification] = [:]

  // These are only the hit if everything else fails.
  public static var fallbackMapping: [String: Specification] = [:]

  public static func specificationForModel(_ name: String) -> Specification? {
    if let override = overrideMapping[name] {
      return override
    }
    return specificationMapping[name] ?? fallbackMapping[name]
  }

  public static func keywordForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return "" }
    return specification.keyword
  }

  public static func tokenLengthForModel(_ name: String) -> Int {
    guard let specification = specificationForModel(name) else { return 0 }
    return specification.length
  }

  public static func versionForModel(_ name: String) -> ModelVersion {
    guard let specification = specificationForModel(name) else { return .v1 }
    return specification.version
  }

  public static func fileSHA256ForModelDownloaded(_ name: String) -> String? {
    return fileSHA256[name]
  }

  public enum Modifier {
    case clipG
    case clipL
    case t5xxl
    case pilet5xl
    case chatglm3_6b
    case llama3
    case qwen25
    case qwen3
    case mistral3
    case gemma3
  }

  public static func embeddingForModel<FloatType: TensorNumeric>(
    _ name: String, graph: DynamicGraph, modifier: Modifier, of: FloatType.Type = FloatType.self
  ) -> Tensor<FloatType>? {
    guard let specification = specificationForModel(name) else { return nil }
    let version = specification.version
    let tokenLength = specification.length
    let count: Int
    switch version {
    case .v1:
      count = 768
    case .v2:
      count = 1024
    case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b:
      switch modifier {
      case .clipG:
        count = 1280
      case .clipL:
        count = 768
      case .t5xxl, .chatglm3_6b, .llama3:
        count = 4096
      case .pilet5xl:
        count = 2048
      case .qwen25:
        count = 3584
      case .qwen3:
        count = 2560
      case .mistral3:
        count = 15360
      case .gemma3:
        count = 3840
      }
    case .pixart:
      count = 4096
    case .auraflow:
      count = 2048
    case .flux1:
      count = 4096
    case .flux2:
      count = 15360
    case .flux2_9b:
      count = 12288
    case .flux2_4b:
      count = 7680
    case .cosmos2_5_2b:
      count = 1024
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b,
      .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
      .ernieImage, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2,
      .longcatVideoAvatar1_5, .minimaxH3:
      fatalError()
    }
    var tensor: Tensor<FloatType>? = nil
    graph.openStore(filePathForModelDownloaded(name), flags: .readOnly) {
      let tensorName: String
      switch version {
      case .v1, .v2, .pixart, .auraflow, .flux1:
        tensorName = "string_to_param"
      case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b:
        switch modifier {
        case .clipG:
          tensorName = "string_to_param_clip_g"
        case .clipL:
          tensorName = "string_to_param_clip_l"
        case .t5xxl:
          tensorName = "string_to_param_t5_xxl"
        case .llama3:
          tensorName = "string_to_param_llama3"
        case .pilet5xl:
          tensorName = "string_to_param_pile_t5_xl"
        case .chatglm3_6b:
          tensorName = "string_to_param_chatglm3_6b"
        case .qwen25:
          tensorName = "string_to_param_qwen25"
        case .qwen3:
          tensorName = "string_to_param_qwen3"
        case .mistral3:
          tensorName = "string_to_param_mistral3"
        case .gemma3:
          tensorName = "string_to_param_gemma3"
        }
      case .cosmos2_5_2b:
        tensorName = "string_to_param_qwen3"
      case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b,
        .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
        .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b,
        .ideogram4, .krea2, .longcatVideoAvatar1_5, .minimaxH3:
        fatalError()
      }
      guard let anyTensor = $0.read(tensorName) else { return }
      tensor = Tensor<FloatType>(from: anyTensor).reshaped(.NC(tokenLength, count)).toCPU()
    }
    return tensor
  }

  public static func availableFiles(excluding file: String?) -> Set<String> {
    var files = Set<String>()
    for specification in availableSpecifications {
      guard specification.file != file, TextualInversionZoo.isModelDownloaded(specification.file)
      else { continue }
      files.insert(specification.file)
    }
    return files
  }
}
