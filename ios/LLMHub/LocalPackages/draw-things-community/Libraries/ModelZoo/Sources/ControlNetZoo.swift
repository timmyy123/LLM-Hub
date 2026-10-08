import Diffusion
import Foundation

public struct ControlNetZoo: DownloadZoo {
  public struct Specification: Codable {
    public var name: String
    public var file: String
    public var modifier: ControlHintType?
    public var version: ModelVersion
    public var type: ControlType
    public var globalAveragePooling: Bool = false
    public var imageEncoder: String? = nil
    public var preprocessor: String? = nil
    public var transformerBlocks: [Int]? = nil
    public var deprecated: Bool? = nil
    public var imageEncoderVersion: ImageEncoderVersion? = nil
    public var ipAdapterConfig: IPAdapterConfig? = nil
    public var autoencoder: String? = nil
    public var note: String? = nil
    public init(
      name: String, file: String, modifier: ControlHintType?, version: ModelVersion,
      type: ControlType, globalAveragePooling: Bool = false, imageEncoder: String? = nil,
      preprocessor: String? = nil, transformerBlocks: [Int]? = nil, deprecated: Bool? = nil,
      imageEncoderVersion: ImageEncoderVersion? = nil, ipAdapterConfig: IPAdapterConfig? = nil,
      autoencoder: String? = nil, note: String? = nil
    ) {
      self.name = name
      self.file = file
      self.modifier = modifier
      self.version = version
      self.type = type
      self.globalAveragePooling = globalAveragePooling
      self.imageEncoder = imageEncoder
      self.preprocessor = preprocessor
      self.transformerBlocks = transformerBlocks
      self.deprecated = deprecated
      self.imageEncoderVersion = imageEncoderVersion
      self.ipAdapterConfig = ipAdapterConfig
      self.autoencoder = autoencoder
      self.note = note
    }
  }

  private static var fileSHA256: [String: String] = [
    "controlnet_canny_1.x_f16.ckpt":
      "4f4dc0d747a5e8650bcde753e01978e263af5b0e29e8f228437645fed63af546",
    "controlnet_canny_1.x_v1.1_f16.ckpt":
      "23a4d6415626e1ee6ae674e2ecf3f116b560f1b7f4fa9f71e78e010c153c2b5c",
    "controlnet_depth_1.x_f16.ckpt":
      "9485159df4ba6319217df02f70b294214af63959ca5d5c7d1642bfb788d75e76",
    "controlnet_depth_1.x_v1.1_f16.ckpt":
      "4bc2a430b78bff34ee6e6a042f564c10d07e36f98371a095383438299164b9ad",
    "controlnet_normal_1.x_f16.ckpt":
      "83c33c16cbf8c3380e3bcaa2ba2206ade513c1ab2092441dee9840e5589b3c86",
    "controlnet_openpose_1.x_f16.ckpt":
      "2ce1b39fea2b42617975dc5b2b27d46b081736b3075938f7474083aa682c025a",
    "controlnet_openpose_1.x_v1.1_f16.ckpt":
      "3b4e1e25aeed4c942c50219b1554ffa69aa6e1c52bfae9489a12b1905f78147e",
    "controlnet_scribble_1.x_f16.ckpt":
      "a80ab8813c44e1c54a52d51e992f371f4525d59b4e287879e721a703351c785e",
    "controlnet_scribble_1.x_v1.1_f16.ckpt":
      "f76364b95ff5121945c103ea9d9d135d689ace75cee9e5a5c83b2d2daaaa73df",
    "controlnet_canny_2.x_f16.ckpt":
      "90ecb4beb7d7f4bf386ab2622c5d551ec72004c21591a8bce890fcca6111f87f",
    "controlnet_depth_2.x_f16.ckpt":
      "1a9bf582c765c134c92dd50389d0a035ed7c7704d435583e928b26f07d0f4d40",
    "controlnet_openpose_2.x_f16.ckpt":
      "6cf4892b67f8faaaa8453855d7eebce254860ffa734a71ddccd03066eac5040f",
    "controlnet_scribble_2.x_f16.ckpt":
      "c6daf0745c3c1d1bfce9549bc5c25e4dd62eb496f3b6b5f5d1ef295e117c7421",
    "t2iadapter_canny_1.x_f16.ckpt":
      "df701a1abff779cfd99d453cf84ccf1f70ae8cc37786d5597b2d14c41b85776f",
    "t2iadapter_depth_1.x_f16.ckpt":
      "288031881f33c2dfe6610f02a155a6494f18965008c2724641d095f1aaff40c8",
    "t2iadapter_sketch_1.x_f16.ckpt":
      "90c702589dc2aa510fac1f8ce4c06ecfa68e6473fb1d5c5e686a823d663b570a",
    "t2iadapter_openpose_1.x_f16.ckpt":
      "9989456179798c5de289b1871febd4d6815c6699ed75f4910ac488fd32331623",
    "t2iadapter_color_1.x_f16.ckpt":
      "46d5ad65eacb34fee35b46eb949ca627e3a9f686cfa1cdee946c30a0896b3ffa",
    "controlnet_lineart_1.x_v1.1_f16.ckpt":
      "51b7216e4c2482fa8c04448199b82796be2de694ad3a6c87b614fca258eaf82e",
    "controlnet_lineart_anime_1.x_v1.1_f16.ckpt":
      "3d05520049decead0cd011be724be0e200176dae8fdf892a1c52931986008123",
    "controlnet_softedge_1.x_v1.1_f16.ckpt":
      "a6aa6cfa270d29aca782955b2128507ebb40143442772fdbbbfa8b745d64edee",
    "controlnet_normalbae_1.x_v1.1_f16.ckpt":
      "3b514fc86d0f1cae94d59d398df26a5ac231384ec42fd5f8871334fac5962208",
    "controlnet_seg_1.x_v1.1_f16.ckpt":
      "e83966350ef6c75771ae02485c692da40b27d434ffa37c15c305a5fb84a0275b",
    "controlnet_mlsd_1.x_v1.1_f16.ckpt":
      "d151172bc4813b9987209909278c398b747375880c6f17ca644677bde29bd915",
    "controlnet_inpaint_1.x_v1.1_f16.ckpt":
      "70286d25c2d04eba6041ee66f6c0f2de57eef4eb0ee77a9d683fc05c69002992",
    "controlnet_ip2p_1.x_v1.1_f16.ckpt":
      "ed73edd4b9a742620456f40240e8ecf7b5aa7f87a464ab46cf933c79b2d36ff0",
    "controlnet_shuffle_1.x_v1.1_f16.ckpt":
      "af49bc48a16334234c380d2fa781a8881fee05d8465c1c4c3f2d4a7fa5b80d3a",
    "controlnet_tile_1.x_v1.1_f16.ckpt":
      "e8af246b048b0b9a77ac934fb0a16f7875e3290adb587331658d5ff93bc4bf1e",
    "controlnet_qr_code_monster_1.x_v2.0_f16.ckpt":
      "6609d305dccdf5ff59b33eacbaf16acddc2ff79388b1c636a9e25bb271fc2ee4",
    "open_clip_vit_h14_vision_model_f16.ckpt":
      "87b70da1898b05bfc7b53e3587fb5c0383fe4f48ccae286e8dde217bbee1e80d",
    "ip_adapter_plus_xl_base_open_clip_h14_f16.ckpt":
      "80e31acd88c4e79a68a5c4452cb24e930c5c9c3efd6afef239b1ef88fb145a17",
    "ip_adapter_plus_face_sd_v1.x_open_clip_h14_f16.ckpt":
      "10377fb1429a8b3d09e9400e71f4601794767e621889a9a69e47ce3b34a2e35a",
    "ip_adapter_plus_sd_v1.x_open_clip_h14_f16.ckpt":
      "c8320d0f14578590017192804ef4616d148089b76699363e7ecc9792b93ae3fb",
    "ip_adapter_plus_face_xl_base_open_clip_h14_f16.ckpt":
      "0c042854036be4a4a89bf155b6af7b997965c9de72452efe36f6b33cd2a9bd31",
    "ip_adapter_full_face_sd_v1.x_open_clip_h14_f16.ckpt":
      "1ddc478f9c9b11930274e89a2d60706d0d38ace6f2c6408117c6b611055e78e7",
    "controlnet_canny_sdxl_v1.0_mid_f16.ckpt":
      "732fc3543b725d93854ef6950291123edfb6056dc372cdcf7adad4df0055dbe8",
    "controlnet_depth_sdxl_v1.0_mid_f16.ckpt":
      "6629c9ccdc2a7741c1100089c28588df7f3d3dd26260c17e9204765d62167215",
    "wan_v2.1_1.3b_vace_480p_f16.ckpt":
      "974eb1bd38ba2dcbf228e821102234634fd57a88d359af3695c4772c179a0e85",
    "wan_v2.1_14b_vace_720p_f16.ckpt":
      "5337d4bfd84a9d446486c8a2fe3e9a81e372952e053e3044b4f377dbe01cc256",
    "wan_v2.1_1.3b_vace_480p_q8p.ckpt":
      "62de000252a97b4c79aefb76caa9a468e6faf59bf2786c2301b8f11be0baf5e8",
    "wan_v2.1_14b_vace_720p_q8p.ckpt":
      "2930e1c3e7c25ae0ec128d5e9889bb07ae2ea23ea7e7900771a4b55befcd7dbd",
  ]

  public static let builtinSpecifications: [Specification] = [
    Specification(
      name: "VACE (Wan 2.1, 1.3B)",
      file: "wan_v2.1_1.3b_vace_480p_f16.ckpt",
      modifier: .shuffle, version: .wan21_1_3b, type: .controlnet),
    Specification(
      name: "VACE (Wan 2.1, 1.3B) (8-bit)",
      file: "wan_v2.1_1.3b_vace_480p_q8p.ckpt",
      modifier: .shuffle, version: .wan21_1_3b, type: .controlnet),
    Specification(
      name: "VACE (Wan 2.1, 14B)",
      file: "wan_v2.1_14b_vace_720p_f16.ckpt",
      modifier: .shuffle, version: .wan21_14b, type: .controlnet),
    Specification(
      name: "VACE (Wan 2.1, 14B) (8-bit)",
      file: "wan_v2.1_14b_vace_720p_q8p.ckpt",
      modifier: .shuffle, version: .wan21_14b, type: .controlnet),
    Specification(
      name: "Canny Edge Map (SD v1.x, ControlNet 1.0)", file: "controlnet_canny_1.x_f16.ckpt",
      modifier: .canny, version: .v1, type: .controlnet, deprecated: true),
    Specification(
      name: "Canny Edge Map (SD v1.x, ControlNet 1.1)", file: "controlnet_canny_1.x_v1.1_f16.ckpt",
      modifier: .canny, version: .v1, type: .controlnet),
    Specification(
      name: "Canny Edge Map (SD v2.x, ControlNet)", file: "controlnet_canny_2.x_f16.ckpt",
      modifier: .canny, version: .v2, type: .controlnet),
    Specification(
      name: "Canny Edge Map (SDXL, ControlNet, Diffusers 1.0 Mid)",
      file: "controlnet_canny_sdxl_v1.0_mid_f16.ckpt",
      modifier: .canny, version: .sdxlBase, type: .controlnet, transformerBlocks: [0, 0, 1, 1]),
    Specification(
      name: "Canny Edge Map (SD v1.x, T2I Adapter)", file: "t2iadapter_canny_1.x_f16.ckpt",
      modifier: .canny, version: .v1, type: .t2iadapter),
    Specification(
      name: "Depth Map (SD v1.x, ControlNet 1.0)", file: "controlnet_depth_1.x_f16.ckpt",
      modifier: .depth, version: .v1, type: .controlnet, deprecated: true),
    Specification(
      name: "Depth Map (SD v1.x, ControlNet 1.1)", file: "controlnet_depth_1.x_v1.1_f16.ckpt",
      modifier: .depth, version: .v1, type: .controlnet),
    Specification(
      name: "Depth Map (SD v2.x, ControlNet)", file: "controlnet_depth_2.x_f16.ckpt",
      modifier: .depth, version: .v2, type: .controlnet),
    Specification(
      name: "Depth Map (SDXL, ControlNet, Diffusers 1.0 Mid)",
      file: "controlnet_depth_sdxl_v1.0_mid_f16.ckpt",
      modifier: .depth, version: .sdxlBase, type: .controlnet, transformerBlocks: [0, 0, 1, 1]),
    Specification(
      name: "Depth Map (SD v1.x, T2I Adapter)", file: "t2iadapter_depth_1.x_f16.ckpt",
      modifier: .depth, version: .v1, type: .t2iadapter),
    Specification(
      name: "Scribble (SD v1.x, ControlNet 1.0)", file: "controlnet_scribble_1.x_f16.ckpt",
      modifier: .scribble, version: .v1, type: .controlnet, preprocessor: "hed_f16.ckpt",
      deprecated: true),
    Specification(
      name: "Scribble (SD v1.x, ControlNet 1.1)", file: "controlnet_scribble_1.x_v1.1_f16.ckpt",
      modifier: .scribble, version: .v1, type: .controlnet, preprocessor: "hed_f16.ckpt"),
    Specification(
      name: "Scribble (SD v2.x, ControlNet)", file: "controlnet_scribble_2.x_f16.ckpt",
      modifier: .scribble, version: .v2, type: .controlnet, preprocessor: "hed_f16.ckpt"),
    Specification(
      name: "Scribble (SD v1.x, T2I Adapter)", file: "t2iadapter_sketch_1.x_f16.ckpt",
      modifier: .scribble, version: .v1, type: .t2iadapter),
    Specification(
      name: "Pose (SD v1.x, ControlNet 1.0)", file: "controlnet_openpose_1.x_f16.ckpt",
      modifier: .pose, version: .v1, type: .controlnet, deprecated: true),
    Specification(
      name: "Pose (SD v1.x, ControlNet 1.1)", file: "controlnet_openpose_1.x_v1.1_f16.ckpt",
      modifier: .pose,
      version: .v1, type: .controlnet),
    Specification(
      name: "Pose (SD v2.x, ControlNet)", file: "controlnet_openpose_2.x_f16.ckpt", modifier: .pose,
      version: .v2, type: .controlnet),
    Specification(
      name: "Pose (SD v1.x, T2I Adapter)", file: "t2iadapter_openpose_1.x_f16.ckpt",
      modifier: .pose, version: .v1, type: .t2iadapter),
    Specification(
      name: "Color Palette (SD v1.x, T2I Adapter)", file: "t2iadapter_color_1.x_f16.ckpt",
      modifier: .color, version: .v1, type: .t2iadapter),
    Specification(
      name: "Normal Map (SD v1.x, ControlNet 1.1)", file: "controlnet_normalbae_1.x_v1.1_f16.ckpt",
      modifier: .normalbae,
      version: .v1, type: .controlnet),
    Specification(
      name: "LineArt (SD v1.x, ControlNet 1.1)", file: "controlnet_lineart_1.x_v1.1_f16.ckpt",
      modifier: .lineart,
      version: .v1, type: .controlnet),
    Specification(
      name: "LineArt Anime (SD v1.x, ControlNet 1.1)",
      file: "controlnet_lineart_anime_1.x_v1.1_f16.ckpt", modifier: .lineart,
      version: .v1, type: .controlnet),
    Specification(
      name: "Soft Edge (SD v1.x, ControlNet 1.1)", file: "controlnet_softedge_1.x_v1.1_f16.ckpt",
      modifier: .softedge, version: .v1, type: .controlnet, preprocessor: "hed_f16.ckpt"),
    Specification(
      name: "Segmentation (SD v1.x, ControlNet 1.1)", file: "controlnet_seg_1.x_v1.1_f16.ckpt",
      modifier: .seg, version: .v1, type: .controlnet),
    Specification(
      name: "Inpainting (SD v1.x, ControlNet 1.1)", file: "controlnet_inpaint_1.x_v1.1_f16.ckpt",
      modifier: .inpaint, version: .v1, type: .controlnet),
    Specification(
      name: "Instruct Pix2Pix (SD v1.x, ControlNet 1.1)", file: "controlnet_ip2p_1.x_v1.1_f16.ckpt",
      modifier: .ip2p, version: .v1, type: .controlnet),
    Specification(
      name: "Shuffle (SD v1.x, ControlNet 1.1)", file: "controlnet_shuffle_1.x_v1.1_f16.ckpt",
      modifier: .shuffle, version: .v1, type: .controlnet, globalAveragePooling: true),
    Specification(
      name: "MLSD Hough Map (SD v1.x, ControlNet 1.1)", file: "controlnet_mlsd_1.x_v1.1_f16.ckpt",
      modifier: .mlsd, version: .v1, type: .controlnet),
    Specification(
      name: "Tile (SD v1.x, ControlNet 1.1)", file: "controlnet_tile_1.x_v1.1_f16.ckpt",
      modifier: .tile, version: .v1, type: .controlnet),
    Specification(
      name: "QR Code (SD v1.x, ControlNet Monster 2.0)",
      file: "controlnet_qr_code_monster_1.x_v2.0_f16.ckpt",
      modifier: .scribble, version: .v1, type: .controlnet),
    Specification(
      name: "IP Adapter Plus (SDXL Base)",
      file: "ip_adapter_plus_xl_base_open_clip_h14_f16.ckpt",
      modifier: .shuffle, version: .sdxlBase, type: .ipadapterplus,
      imageEncoder: "open_clip_vit_h14_vision_model_f16.ckpt"),
    Specification(
      name: "IP Adapter Plus Face (SDXL Base)",
      file: "ip_adapter_plus_face_xl_base_open_clip_h14_f16.ckpt",
      modifier: .shuffle, version: .sdxlBase, type: .ipadapterplus,
      imageEncoder: "open_clip_vit_h14_vision_model_f16.ckpt"),
    Specification(
      name: "IP Adapter Plus (SD v1.x)",
      file: "ip_adapter_plus_sd_v1.x_open_clip_h14_f16.ckpt",
      modifier: .shuffle, version: .v1, type: .ipadapterplus,
      imageEncoder: "open_clip_vit_h14_vision_model_f16.ckpt"),
    Specification(
      name: "IP Adapter Plus Face (SD v1.x)",
      file: "ip_adapter_plus_face_sd_v1.x_open_clip_h14_f16.ckpt",
      modifier: .shuffle, version: .v1, type: .ipadapterplus,
      imageEncoder: "open_clip_vit_h14_vision_model_f16.ckpt"),
    Specification(
      name: "IP Adapter Full Face (SD v1.x)",
      file: "ip_adapter_full_face_sd_v1.x_open_clip_h14_f16.ckpt",
      modifier: .shuffle, version: .v1, type: .ipadapterfull,
      imageEncoder: "open_clip_vit_h14_vision_model_f16.ckpt"),
  ]

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in availableSpecifications {
      mapping[specification.file] = specification
    }
    return mapping
  }()

  public static func appendCustomSpecification(_ specification: Specification) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = filePathForModelDownloaded("custom_controlnet.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      if let jsonSpecification = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value }) {
        customSpecifications.append(contentsOf: jsonSpecification)
      }
    }
    if let firstIndex = (customSpecifications.firstIndex { $0.name == specification.name }) {
      customSpecifications[firstIndex] = specification
    } else {
      customSpecifications.append(specification)
    }
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    guard let jsonData = try? jsonEncoder.encode(customSpecifications) else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)
    // Modify these two are not thread safe. availableSpecifications are OK. specificationMapping is particularly problematic (as it is access on both main thread and a background thread).
    if let firstIndex = (availableSpecifications.firstIndex { $0.name == specification.name }) {
      availableSpecifications[firstIndex] = specification
    } else {
      availableSpecifications.append(specification)
    }
    specificationMapping[specification.file] = specification
  }

  public static func sortCustomSpecifications() {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = filePathForModelDownloaded("custom_controlnet.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      if let jsonSpecification = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value }) {
        customSpecifications.append(contentsOf: jsonSpecification)
      }
    }
    customSpecifications = customSpecifications.sorted(by: {
      $0.name.localizedStandardCompare($1.name) == .orderedAscending
    })

    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    guard let jsonData = try? jsonEncoder.encode(customSpecifications) else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)

    // Because this only does sorting, it won't impact the builtinModels set.
    var availableSpecifications = builtinSpecifications
    let builtinModels = Set(builtinSpecifications.map { $0.file })
    for specification in customSpecifications {
      if builtinModels.contains(specification.file) {
        availableSpecifications = availableSpecifications.filter { $0.file != specification.file }
      }
      availableSpecifications.append(specification)
    }
    self.availableSpecifications = availableSpecifications
  }

  public static func filePathForModelDownloaded(_ name: String) -> String {
    return ModelZoo.filePathForModelDownloaded(name)
  }

  public static func isModelDownloaded(_ name: String, memorizedBy: Set<String>) -> Bool {
    return ModelZoo.isModelDownloaded(name, memorizedBy: memorizedBy)
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

  public static func filesToDownload(_ specification: Specification)
    -> [(name: String, subtitle: String, file: String, sha256: String?)]
  {
    var filesToDownload = [(name: String, subtitle: String, file: String, sha256: String?)]()
    let name = specification.name
    let version = ModelZoo.humanReadableNameForVersion(specification.version)
    filesToDownload.append((name: name, subtitle: version, file: specification.file, sha256: nil))
    if let imageEncoder = specification.imageEncoder {
      filesToDownload.append((name: name, subtitle: version, file: imageEncoder, sha256: nil))
    }
    if let autoencoder = specification.autoencoder {
      filesToDownload.append((name: name, subtitle: version, file: autoencoder, sha256: nil))
    }
    if let preprocessor = specification.preprocessor {
      filesToDownload.append((name: name, subtitle: version, file: preprocessor, sha256: nil))
    }
    if specification.modifier == .depth {
      let name = EverythingZoo.humanReadableNameForModel("depth_anything_v2.0_f16.ckpt")
      filesToDownload.append(
        (name: name, subtitle: "", file: "depth_anything_v2.0_f16.ckpt", sha256: nil))
    }
    return filesToDownload
  }

  public static func isModelDownloaded(
    _ specification: Specification, memorizedBy: Set<String> = []
  ) -> Bool {
    // Make sure both the model and the image encoder is downloaded.
    return ModelZoo.isModelDownloaded(specification.file, memorizedBy: memorizedBy)
      && (specification.imageEncoder.map {
        ModelZoo.isModelDownloaded($0, memorizedBy: memorizedBy)
      } ?? true)
      && (specification.autoencoder.map { ModelZoo.isModelDownloaded($0, memorizedBy: memorizedBy) }
        ?? true)
      && (specification.preprocessor.map {
        ModelZoo.isModelDownloaded($0, memorizedBy: memorizedBy)
      } ?? true)
  }

  public static func humanReadableNameForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return name }
    return specification.name
  }

  public static func modifierForModel(_ name: String) -> ControlHintType? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.modifier
  }

  public static func versionForModel(_ name: String) -> ModelVersion {
    guard let specification = specificationForModel(name) else { return .v1 }
    return specification.version
  }

  public static func imageEncoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.imageEncoder
  }

  public static func imageEncoderVersionForModel(_ name: String) -> ImageEncoderVersion {
    guard let specification = specificationForModel(name) else { return .openClipH14 }
    return specification.imageEncoderVersion ?? .openClipH14
  }

  public static func IPAdapterConfigForModel(_ name: String) -> IPAdapterConfig? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.ipAdapterConfig
  }

  public static func autoencoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.autoencoder
  }

  public static func preprocessorForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.preprocessor
  }

  public static func typeForModel(_ name: String) -> ControlType {
    guard let specification = specificationForModel(name) else { return .controlnet }
    return specification.type
  }

  public static func globalAveragePoolingForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.globalAveragePooling
  }

  public static func transformerBlocksForModel(_ name: String) -> [Int] {
    guard let specification = specificationForModel(name) else { return [] }
    return specification.transformerBlocks ?? []
  }

  public static func noteForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return "" }
    return specification.note ?? ""
  }

  public static func fileSHA256ForModelDownloaded(_ name: String) -> String? {
    return fileSHA256[name]
  }

  private static let builtinModelsAndAvailableSpecifications: (Set<String>, [Specification]) = {
    let jsonFile = filePathForModelDownloaded("custom_controlnet.json")
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

  public static func isBuiltinControl(_ name: String) -> Bool {
    return builtinModels.contains(name)
  }

  public static func mergeFileSHA256(_ sha256: [String: String]) {
    var fileSHA256 = fileSHA256
    for (key, value) in sha256 {
      fileSHA256[key] = value
    }
    self.fileSHA256 = fileSHA256
  }

  public static func isModelDeprecated(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.deprecated ?? false
  }

  public static func availableFiles(excluding file: String?) -> Set<String> {
    var files = Set<String>()
    for specification in availableSpecifications {
      guard specification.file != file, ControlNetZoo.isModelDownloaded(specification.file) else {
        continue
      }
      files.insert(specification.file)
      if let imageEncoder = specification.imageEncoder {
        files.insert(imageEncoder)
      }
      if let autoencoder = specification.autoencoder {
        files.insert(autoencoder)
      }
      if let preprocessor = specification.preprocessor {
        files.insert(preprocessor)
      }
    }
    return files
  }
}
