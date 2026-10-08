import DataModels
import Diffusion
import Foundation

public struct FailableDecodable<T: Decodable>: Decodable {
  public let value: T?
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    value = try? container.decode(T.self)
  }
}

public struct ModelZoo: DownloadZoo {
  public static func humanReadableNameForVersion(_ version: ModelVersion) -> String {
    switch version {
    case .minimaxH3:
      return "MiniMax H3"
    case .longcatVideoAvatar1_5:
      return "LongCat-Video Avatar 1.5"
    case .v1:
      return "Stable Diffusion v1"
    case .v2:
      return "Stable Diffusion v2"
    case .kandinsky21:
      return "Kandinsky v2.1"
    case .sdxlBase:
      return "Stable Diffusion XL Base"
    case .sdxlRefiner:
      return "Stable Diffusion XL Refiner"
    case .ssd1b:
      return "Segmind Stable Diffusion XL 1B"
    case .svdI2v:
      return "Stable Video Diffusion"
    case .wurstchenStageC, .wurstchenStageB:
      return "Stable Cascade (Würstchen v3.0)"
    case .sd3:
      return "Stable Diffusion 3 Medium"
    case .sd3Large:
      return "Stable Diffusion 3 Large"
    case .pixart:
      return "PixArt Sigma"
    case .auraflow:
      return "AuraFlow"
    case .flux1:
      return "FLUX.1"
    case .hunyuanVideo:
      return "Hunyuan Video"
    case .wan21_1_3b:
      return "Wan v2.1 1.3B"
    case .wan21_14b:
      return "Wan v2.x 14B"
    case .hiDreamI1:
      return "HiDream I1"
    case .hiDreamO1:
      return "HiDream O1"
    case .qwenImage2_1:
      return "Qwen Image 2.1"
    case .qwenImage:
      return "Qwen Image"
    case .wan22_5b:
      return "Wan v2.2 5B"
    case .zImage:
      return "Z Image"
    case .ernieImage:
      return "ERNIE Image"
    case .flux2:
      return "FLUX.2"
    case .flux2_9b:
      return "FLUX.2 9B"
    case .flux2_4b:
      return "FLUX.2 4B"
    case .cosmos2_5_2b:
      return "Cosmos 2.5 2B"
    case .ltx2:
      return "LTX-2"
    case .ltx2_3:
      return "LTX-2.3"
    case .seedvr2_3b, .seedvr2_7b:
      return "SeedVR2"
    case .ideogram4:
      return "Ideogram 4"
    case .krea2:
      return "Krea 2"
    }
  }

  public enum NoiseDiscretization: Codable {
    case edm(Denoiser.Parameterization.EDM)
    case ddpm(Denoiser.Parameterization.DDPM)
    case rf(Denoiser.Parameterization.RF)
  }

  public struct Specification: Codable {
    public struct MMDiT: Codable {
      public var qkNorm: Bool
      public var dualAttentionLayers: [Int]
      public var distilledGuidanceLayers: Int?
      public var activationQkScaling: [Int: Int]?
      public var activationProjScaling: [Int: Int]?
      public var activationFfnProjUpScaling: [Int: Int]?
      public var activationFfnScaling: [Int: Int]?
      public init(
        qkNorm: Bool, dualAttentionLayers: [Int], distilledGuidanceLayers: Int? = nil,
        activationQkScaling: [Int: Int]? = nil, activationProjScaling: [Int: Int]? = nil,
        activationFfnProjUpScaling: [Int: Int]? = nil, activationFfnScaling: [Int: Int]? = nil
      ) {
        self.qkNorm = qkNorm
        self.dualAttentionLayers = dualAttentionLayers
        self.distilledGuidanceLayers = distilledGuidanceLayers
        self.activationQkScaling = activationQkScaling
        self.activationProjScaling = activationProjScaling
        self.activationFfnProjUpScaling = activationFfnProjUpScaling
        self.activationFfnScaling = activationFfnScaling
      }
    }

    public struct RemoteApiModelConfig: Codable {
      public enum JSONNumber: Codable {
        case integer(Int)
        case fraction(Double)

        public init(from decoder: Decoder) throws {
          let container = try decoder.singleValueContainer()
          if let doubleValue = try? container.decode(Double.self) {
            self = .fraction(doubleValue)
          } else if let integerValue = try? container.decode(Int.self) {
            self = .integer(integerValue)
          } else {
            throw DecodingError.dataCorruptedError(
              in: container, debugDescription: "Cannot decode value")
          }
        }

        public func encode(to encoder: Encoder) throws {
          var container = encoder.singleValueContainer()
          switch self {
          case .integer(let value): try container.encode(value)
          case .fraction(let value): try container.encode(value)
          }
        }

        public var value: Any {
          switch self {
          case .fraction(let value): return value
          case .integer(let value): return value
          }
        }
      }

      public enum JSONValue: Codable {
        case string(String)
        case number(JSONNumber)
        case bool(Bool)

        public init(from decoder: Decoder) throws {
          let container = try decoder.singleValueContainer()
          if let stringValue = try? container.decode(String.self) {
            self = .string(stringValue)
          } else if let numberValue = try? container.decode(JSONNumber.self) {
            self = .number(numberValue)
          } else if let boolValue = try? container.decode(Bool.self) {
            self = .bool(boolValue)
          } else {
            throw DecodingError.dataCorruptedError(
              in: container, debugDescription: "Cannot decode value")
          }
        }

        public func encode(to encoder: Encoder) throws {
          var container = encoder.singleValueContainer()
          switch self {
          case .string(let value): try container.encode(value)
          case .number(let value): try container.encode(value)
          case .bool(let value): try container.encode(value)
          }
        }
        public var value: Any {
          switch self {
          case .string(let value): return value
          case .number(let number): return number.value
          case .bool(let value): return value
          }
        }
      }
      public indirect enum CustomRequestBodyValue: Codable {
        case string(String)
        case dictionary([String: CustomRequestBodyValue])

        public init(from decoder: Decoder) throws {
          let container = try decoder.singleValueContainer()
          if let stringValue = try? container.decode(String.self) {
            self = .string(stringValue)
          } else if let dictValue = try? container.decode([String: CustomRequestBodyValue].self) {
            self = .dictionary(dictValue)
          } else {
            throw DecodingError.dataCorruptedError(
              in: container, debugDescription: "Cannot decode CustomRequestBodyValue")
          }
        }

        public func encode(to encoder: Encoder) throws {
          var container = encoder.singleValueContainer()
          switch self {
          case .string(let value): try container.encode(value)
          case .dictionary(let value): try container.encode(value)
          }
        }
      }

      public enum ApiFileFormat: String, Codable {
        case image = "image"
        case video = "video"
      }

      public enum ResultPath: Codable {
        case base64(path: String)
        case url(path: String)

        public var value: String {
          switch self {
          case .base64(let value): return value
          case .url(let value): return value
          }
        }
      }

      public var endpoint: String
      public var url: String
      public var remoteApiModelConfigMapping: [String: String]

      public var ephemeralApiSecret: Bool
      public var requestType: String
      public var taskIdPath: String
      public var statusUrlTemplate: String
      public var resultPath: ResultPath
      public var statusPath: String
      public var successStatus: String
      public var failureStatus: String
      public var pendingStatuses: [String]
      public var errorMsgPath: String?
      public var apiKey: String?
      public var apiSecret: String?
      public var apiFileFormat: ApiFileFormat
      public var pollingInterval: TimeInterval
      public var passthroughConfigs: [String: JSONValue]?
      public var settingsSections: [String]
      public var customImageSizeRatios: [String]?
      public var tokenConfig: [String: String]?
      public var customRequestBody: [String: CustomRequestBodyValue]?
      public var downloadUrlSuffix: String?
      public init(
        endpoint: String,
        url: String,
        remoteApiModelConfigMapping: [String: String],
        ephemeralApiSecret: Bool,
        requestType: String,
        taskIdPath: String,
        statusUrlTemplate: String,
        resultPath: ResultPath,
        statusPath: String,
        successStatus: String,
        failureStatus: String,
        pendingStatuses: [String],
        errorMsgPath: String? = nil,
        apiKey: String? = nil,
        apiSecret: String? = nil,
        apiFileFormat: ApiFileFormat,
        pollingInterval: TimeInterval = 5.0,
        passthroughConfigs: [String: JSONValue]? = nil,
        settingsSections: [String],
        customImageSizeRatios: [String]? = nil,
        tokenConfig: [String: String]? = nil,
        customRequestBody: [String: CustomRequestBodyValue]? = nil,
        downloadUrlSuffix: String? = nil
      ) {
        self.endpoint = endpoint
        self.url = url
        self.remoteApiModelConfigMapping = remoteApiModelConfigMapping
        self.ephemeralApiSecret = ephemeralApiSecret
        self.requestType = requestType
        self.taskIdPath = taskIdPath
        self.statusUrlTemplate = statusUrlTemplate
        self.resultPath = resultPath
        self.statusPath = statusPath
        self.successStatus = successStatus
        self.failureStatus = failureStatus
        self.pendingStatuses = pendingStatuses
        self.errorMsgPath = errorMsgPath
        self.apiSecret = apiSecret
        self.apiKey = apiKey
        self.apiFileFormat = apiFileFormat
        self.pollingInterval = pollingInterval
        self.settingsSections = settingsSections
        self.passthroughConfigs = passthroughConfigs
        self.customImageSizeRatios = customImageSizeRatios
        self.tokenConfig = tokenConfig
        self.customRequestBody = customRequestBody
        self.downloadUrlSuffix = downloadUrlSuffix
      }
    }
    public struct LatentsUpscaler: Codable {
      public enum Scale: String, Codable {
        case x1_5 = "1.5"
        case x2 = "2"
      }
      public var file: String
      public var scale: Scale
    }
    public var name: String
    public var file: String
    public var prefix: String
    public var version: ModelVersion
    public var upcastAttention: Bool
    public var defaultScale: UInt16
    public var textEncoder: String?
    public var autoencoder: String?
    public var modifier: SamplerModifier?
    public var deprecated: Bool?
    public var imageEncoder: String?
    public var clipEncoder: String?
    public var additionalClipEncoders: [String]?
    public var t5Encoder: String?
    public var diffusionMapping: String?
    public var highPrecisionAutoencoder: Bool?
    public var defaultRefiner: String?
    public var isConsistencyModel: Bool?
    public var conditioning: Denoiser.Conditioning?
    public var objective: Denoiser.Objective?
    public var noiseDiscretization: NoiseDiscretization?
    public var latentsMean: [Float]?
    public var latentsStd: [Float]?
    public var audioLatentsMean: [Float]?
    public var audioLatentsStd: [Float]?
    public var latentsScalingFactor: Float?
    public var noiseScalingFactor: Float?
    public var stageModels: [String]?
    public var textEncoderVersion: TextEncoderVersion?
    public var guidanceEmbed: Bool?
    public var paddedTextEncodingLength: Int?
    public var hiresFixScale: UInt16?
    public var mmdit: MMDiT?
    public var builtinLora: Bool?
    public var teaCacheCoefficients: [Float]?
    public var remoteApiModelConfig: RemoteApiModelConfig?
    public var framesPerSecond: Double?
    public var isBf16: Bool?
    public var note: String?
    public var copyright: String?
    public var huggingFaceLink: String?
    public var latentsUpscalers: [LatentsUpscaler]?
    public init(
      name: String, file: String, prefix: String, version: ModelVersion,
      upcastAttention: Bool = false, defaultScale: UInt16 = 8, textEncoder: String? = nil,
      autoencoder: String? = nil, modifier: SamplerModifier? = nil, deprecated: Bool? = nil,
      imageEncoder: String? = nil, clipEncoder: String? = nil,
      additionalClipEncoders: [String]? = nil, t5Encoder: String? = nil,
      diffusionMapping: String? = nil, highPrecisionAutoencoder: Bool? = nil,
      defaultRefiner: String? = nil, isConsistencyModel: Bool? = nil,
      conditioning: Denoiser.Conditioning? = nil, objective: Denoiser.Objective? = nil,
      noiseDiscretization: NoiseDiscretization? = nil, latentsMean: [Float]? = nil,
      latentsStd: [Float]? = nil, audioLatentsMean: [Float]? = nil, audioLatentsStd: [Float]? = nil,
      latentsScalingFactor: Float? = nil, noiseScalingFactor: Float? = nil,
      stageModels: [String]? = nil,
      textEncoderVersion: TextEncoderVersion? = nil, guidanceEmbed: Bool? = nil,
      paddedTextEncodingLength: Int? = nil, hiresFixScale: UInt16? = nil, mmdit: MMDiT? = nil,
      builtinLora: Bool? = nil, teaCacheCoefficients: [Float]? = nil,
      framesPerSecond: Double? = nil, isBf16: Bool? = nil,
      latentsUpscalers: [LatentsUpscaler]? = nil,
      remoteApiModelConfig: RemoteApiModelConfig? = nil, note: String? = nil,
      copyright: String? = nil, huggingFaceLink: String? = nil
    ) {
      self.name = name
      self.file = file
      self.prefix = prefix
      self.version = version
      self.upcastAttention = upcastAttention
      self.defaultScale = defaultScale
      self.textEncoder = textEncoder
      self.autoencoder = autoencoder
      self.modifier = modifier
      self.deprecated = deprecated
      self.imageEncoder = imageEncoder
      self.clipEncoder = clipEncoder
      self.additionalClipEncoders = additionalClipEncoders
      self.t5Encoder = t5Encoder
      self.diffusionMapping = diffusionMapping
      self.highPrecisionAutoencoder = highPrecisionAutoencoder
      self.defaultRefiner = defaultRefiner
      self.isConsistencyModel = isConsistencyModel
      self.conditioning = conditioning
      self.objective = objective
      self.noiseDiscretization = noiseDiscretization
      self.latentsMean = latentsMean
      self.latentsStd = latentsStd
      self.audioLatentsMean = audioLatentsMean
      self.audioLatentsStd = audioLatentsStd
      self.latentsScalingFactor = latentsScalingFactor
      self.noiseScalingFactor = noiseScalingFactor
      self.stageModels = stageModels
      self.textEncoderVersion = textEncoderVersion
      self.guidanceEmbed = guidanceEmbed
      self.paddedTextEncodingLength = paddedTextEncodingLength
      self.hiresFixScale = hiresFixScale
      self.mmdit = mmdit
      self.builtinLora = builtinLora
      self.teaCacheCoefficients = teaCacheCoefficients
      self.remoteApiModelConfig = remoteApiModelConfig
      self.framesPerSecond = framesPerSecond
      self.isBf16 = isBf16
      self.latentsUpscalers = latentsUpscalers
      self.note = note
      self.copyright = copyright
      self.huggingFaceLink = huggingFaceLink
    }
    fileprivate var predictV: Bool? = nil
  }

  private static var fileSHA256: [String: String] = [
    "clip_vit_l14_f16.ckpt": "809bfd12c8d4b3d79c14e850b99130a70854f6fd8dedcacdf429417c02fa3007",
    "open_clip_vit_h14_f16.ckpt":
      "cdaa1b93cb099d4aff8831ba248780cebbb54bcd2810dd242513c4a8c70ba577",
    "sd_v1.4_f16.ckpt": "0e0d62f677aba5aae59d977e8a48b2ad87b6d47d519e92f11b7f988c882e5910",
    "vae_ft_mse_840000_f16.ckpt":
      "3b35514e11dd2b913e0579089babc1dfbd36589a77044c2e9b8065187e2f4154",
    "sd_v1.5_f16.ckpt": "bf867591702e4c5d86cb126a3601d7e494180cce956b8dfaf90e5093d2e7c0f6",
    "sd_v1.5_inpainting_f16.ckpt":
      "4e935e18e3d1be94378d96e0d9cb347fcd75de4821ff1d142c60640313b60ab2",
    "sd_v2.0_f16.ckpt": "73cbc76b4ecc4a8c33bf4c452d396c86c42c2f50361745bd649a98e9ea269a3b",
    "sd_v2.1_f16.ckpt": "2d9a7302668bacf3b801327bc23b116f24a441e6229cc4a4b7c39aaa4bf3c9f7",
    "sd_v2.0_inpainting_f16.ckpt":
      "d42b44d3614a0e22195aa5e4f94f417c7c755a99c463e8730ad8f7071c2c5a92",
    "sd_v2.0_depth_f16.ckpt": "64f907b7bf40954477439dda42dcf2cf864526b4c498279cd4274bce12fe896d",
    "sd_v2.0_768_v_f16.ckpt": "992be2b0b34e0a591b043a07b4fc32bf04210424872230a15169e68ef45cde43",
    "sd_v2.1_768_v_f16.ckpt": "04378818798ab37ce9adc189ea28c342d9edde8511194bf5a205f56bb38cf05c",
    "minisd_v1.4_f16.ckpt": "7aed73bf40b49083be32791de39e192f6ac4aa20fbc98e13d4cdca7b5bdd07bf",
    "wd_v1.3_f16.ckpt": "b6862eec82ec14cdb754c5df5c131631bae5e4664b5622a615629c42e7a43c05",
    "classicanim_v1_f16.ckpt": "168799472175b77492814ae3cf5e9f793a3d3d009592a9e5b03781626ea25590",
    "modi_v1_f16.ckpt": "ca76d84c1783ef367201e4eac2e1dddbce0c40afc6de62a229b80cb04ae7c4f0",
    "arcane_v3_f16.ckpt": "4c55d2239e1f0ff40cc6e1ae518737735f6d1b7613f8f7aca9239205f0be729a",
    "cyberpunk_anime_f16.ckpt": "df55b6c66704b51921e31711adaab9e37bd78fc10733fcd89e6f86426230ef41",
    "redshift_v1_f16.ckpt": "a7fc94bac178414d7caf844787afcaf8c6c273ebf9011fed75703de7839fc257",
    "redshift_768_v_f16.ckpt": "aa6520ae1fc447082230b2eb646c40e6f776f257c453134d0f064a89ac1de751",
    "redshift_768_v_open_clip_vit_h14_f16.ckpt":
      "9c7f1a65fe890f288c2d2ff7cef11b502bf10965a7eaa7d0d43362cab9f90eca",
    "dnd_30000_f16.ckpt": "3de9309cf4541168fb39d96eaba760146b42e7e9870a3096eb4cd097384ea1d9",
    "trnlgcy_f16.ckpt": "3ed86762dda66f5dc728ee1f67085d2ba9f3e3ea1b5b3464b8f3a791954cfa3c",
    "classicanim_v1_clip_vit_l14_f16.ckpt":
      "77cfbb6054a2a5581873c3b1be8c6457bed526d1f15d6cffb6e381401692a488",
    "modi_v1_clip_vit_l14_f16.ckpt":
      "e7907cbb2f7656bb2f6fb4ead4fcb030721e4218ca2a105976b88bce852f2860",
    "arcane_v3_clip_vit_l14_f16.ckpt":
      "954f1e1fb690dcb1820adaf83099b39057e2b1bcbbdc12ecfe37ac17bcad6fa7",
    "cyberpunk_anime_clip_vit_l14_f16.ckpt":
      "d62bb1de4b579d73111b3355cad72b1d8f3bf22519c4bfd1a224bdd952cd0279",
    "redshift_v1_clip_vit_l14_f16.ckpt":
      "95532a3275a81d909d657c98b73ef576809254e29052aaa809d9336c13f182a1",
    "dnd_30000_clip_vit_l14_f16.ckpt":
      "96c75d1c11030a51aa8dac5410cc6fa98b071b52f0f79a07097df022b20754dc",
    "trnlgcy_clip_vit_l14_f16.ckpt":
      "a99adbecbed4e370abcffc2574fac8e664a2530531fdc89b71d2f15711f40545",
    "mdjrny_v4_f16.ckpt": "a0d976948c18943f1281268cc3edbe1d1fa2a4098b5a290d9947a1a299976699",
    "mdjrny_v4_clip_vit_l14_f16.ckpt":
      "ad4e3d64c0a5e81d529c984dcfbdc6858d73e14ebe8788975e6b8c4fbfc17629",
    "nitro_v1_f16.ckpt": "2549d7220cce7f53311fe145878e1af8bcd52efaf15bcb81a2681c0abcddd6c3",
    "nitro_v1_clip_vit_l14_f16.ckpt":
      "2b5424697630a50ed2d1b8c2449e3fb5f613a6569d72d16dc59d1e28a8a0c07d",
    "anything_v3_f16.ckpt": "f4354727512d6b6a2d5e4cf783fdc8475e7981c50b9f387bc93317c22299e505",
    "anything_v3_clip_vit_l14_f16.ckpt":
      "5f1311561bdac6d43e4b3bacbee8c257bf788e6c86b3c69c68247a9abab1050d",
    "anything_v3_vae_f16.ckpt": "3b7d16260a7d211416739285f97d53354b332cfceadb2b7191817f4e1cfb5d57",
    "hassanblend_v1.4_f16.ckpt": "e3566b98cfa81660cd4833c01cd9a05a853e367726d840a5eb16098b53c042ae",
    "lvngvncnt_v2_f16.ckpt": "dbacd01fb82501895afde1bbcf3f16eefeea8043fa3463de640c09a9315460be",
    "lvngvncnt_v2_clip_vit_l14_f16.ckpt":
      "cbdaae485f60c7cb395e5725dd16e769816274b74498d0c45048962a49cc4a06",
    "spiderverse_v1_f16.ckpt": "8c8c80add2d663732e314c3a2fb49c1f2bd98f48190b79227d660ce687516b2d",
    "spiderverse_v1_clip_vit_l14_f16.ckpt":
      "bab7fcf0e615154ff91c88a8fbf9b18a548e8ba0a338fb030a3fedf17ce0602d",
    "eldenring_v3_f16.ckpt": "c6b79886e426d654c9e84cf53a7dd572fbb9e7083c47384a76d02702c54c50c3",
    "eldenring_v3_clip_vit_l14_f16.ckpt":
      "dcd2234e90f8df2c4eb706f665fa860ad54df2ae109cfcd8b235c1c420bd2d4d",
    "papercut_v1_f16.ckpt": "7b1d14757e1c58b1bef55220d0fd10ab4ad8e2670bb4e065e4b6c4e0b6a6395e",
    "papercut_v1_clip_vit_l14_f16.ckpt":
      "bc0c471e51bbe0649922dad862019b96e68d4abf724998bbfa9495e70bd2023d",
    "voxelart_v1_f16.ckpt": "e771d7acd484162377c62a6033b632ea290d4477bf3cb017a750f15ab5350ca7",
    "voxelart_v1_clip_vit_l14_f16.ckpt":
      "ebd3ce92b9ec831a6f217c42be0b8da887484867583156d0e2ceb3e48bae3be8",
    "balloonart_v1_f16.ckpt": "f73bcbd3a6db0dca10afb081a2066a7aea5b117962bd26efc37320dfc3b9b759",
    "balloonart_v1_clip_vit_l14_f16.ckpt":
      "6be250d1c38325f7ee80f3fcd99e1a490f28deb29a8f78188813e8157f1949b3",
    "f222_f16.ckpt": "ae19854df68e232c3bbda8b9322e9f56ccd2d96517a31a82694c40382003f8ae",
    "supermarionation_v2_f16.ckpt":
      "70e13769ee9c8b8c4d4b8240f23b8d8fcef154325fd9162174b75f67c5629440",
    "supermarionation_v2_clip_vit_l14_f16.ckpt":
      "e2da78a79ee90fe352e465326e2dc0c055888c27a84d465cfd9ea2987a83a131",
    "inkpunk_v2_f16.ckpt": "8957387975caf8c56caa6c4c2b9d8fff07bda7a8a2aadec840be3fd623d1d2fe",
    "inkpunk_v2_clip_vit_l14_f16.ckpt":
      "569d9796b5f3b33ed1ce65b27fa3fb4dfdb8ef2440555fa33f30fa8d118cc293",
    "samdoesart_v3_f16.ckpt": "5a55df0470437ac0f3f0c05d77098c6eb8577c61ce0e1b2dc898240fb49fd10e",
    "samdoesart_v3_clip_vit_l14_f16.ckpt":
      "6d84e79c05f9c89172f4b82821a7c8223d3bd6bacfd80934dd85dce71a8f2519",
    "ghibli_v1_f16.ckpt": "dfcf9358528e8892f82b4ba3d0c9245be928e2e920e746383bdaf1b9a3a93151",
    "ghibli_v1_clip_vit_l14_f16.ckpt":
      "bf7c353e5b2b34bff2216742e114ee707f0ad023cc0bfd5ebde779b3b3162a02",
    "analog_v1_f16.ckpt": "ffed9bb928a20f90f9881ac0d51e918c1580562f099fdd45c061c292dec63ab5",
    "analog_v1_clip_vit_l14_f16.ckpt":
      "f144ac4ad344c82c3b1dc69e46aba8d9c6bc20d24de9e48105a3db3e4437108d",
    "dnd_classes_and_species_f16.ckpt":
      "a6059246c1c06edc73646c77a1aa819ca641e0d8ceba0e25365938ab71311174",
    "dnd_classes_and_species_clip_vit_l14_f16.ckpt":
      "09fdf2d991591947e2743e8431e9d6eaf99fe2f524de9c752ebb7a4289225b02",
    "aloeveras_simpmaker_3k1_f16.ckpt":
      "562db3b5ca4961eed207e308073d99293d27f23f90e09dba58f2eb828a2f8e0c",
    "hna_3dkx_1.0b_f16.ckpt":
      "5e9246ff45380d6e0bd22506d559e2d6616b7aa0e42052a92c0270b89de2defa",
    "hna_3dkx_1.0b_clip_vit_l14_f16.ckpt":
      "7317f067a71f1e2a2a886c60574bb391bf31a559b4daa4901c45d1d5d2acc7d6",
    "seek_art_mega_v1_f16.ckpt":
      "0f10cfa16950fc5bb0a31b9974275c256c1a11f26f92ac26be6f7ea91e7019ac",
    "seek_art_mega_v1_clip_vit_l14_f16.ckpt":
      "9dd3af747d71b10d318b876a9285f8cc7c350806585146a3eaa660bcaf54bc7e",
    "instruct_pix2pix_22000_f16.ckpt":
      "ffe6548ff4e803c64f8ca2b84024058e88494329acff29583fbb9f45305dd410",
    "hassanblend_v1.5.1.2_f16.ckpt":
      "e5eb4e11fa1f882dc084a0e061abf6b7f5e7dd11c416ff14842c049b9727c5d1",
    "hassanblend_v1.5.1.2_clip_vit_l14_f16.ckpt":
      "0d572f5e379c48c88aa7ca1d6aff095d94cacaf8b90f6444f4af46a7d3d18f33",
    "hna_3dkx_1.1_f16.ckpt":
      "9e333094d9b73db3e0438f7520c0cd5deb2f0f6b3aa890ce464050cc7dd8d693",
    "hna_3dkx_1.1_clip_vit_l14_f16.ckpt":
      "5ce38e05ada7ec4488c600bc026db1386fb4cdca2882fe51561c49a1bc70da4d",
    "kandinsky_f16.ckpt":
      "563cbf6dd08c81063c45310a7a420b75004d6f226eb7e045f167d03d485fc36a",
    "kandinsky_diffusion_mapping_f16.ckpt":
      "6467fd6ac08bc4d851ed09286f2273f134fe5d6763086ef06551f1285de059f0",
    "kandinsky_movq_f16.ckpt":
      "f7ac86bd2f1b3bb7487a064df64e39fbf92905e40ebfbe943c3859ff05204071",
    "xlm_roberta_f16.ckpt":
      "772cd148b7254d16cd934aad380362cde8869edb34f787eb7cc4776a64e3d5a2",
    "image_vit_l14_f16.ckpt":
      "f75c2ac4b5f8e0c59001ce05ecf5b11ee893f7687b2154075c2ddd7c11fe9b32",
    "deliberate_v2_q6p_q8p.ckpt":
      "4441ea31f748a5af97021747bc457e78ae0c8d632f819a26cb8019610972c0f0",
    "deliberate_v2_clip_vit_l14_f16.ckpt":
      "79dc846fe47f4bd5188bce108c9791f36cc2927bed6f96c8dc7369b345539d81",
    "disney_pixar_cartoon_type_b_q6p_q8p.ckpt":
      "31f38b788e1acdde65288f1e3780c64df9c98cd5fa7fa38bce5bce085f633d95",
    "disney_pixar_cartoon_type_b_clip_vit_l14_f16.ckpt":
      "0401d93b66dff9de82521765bbcb2292904a247e22155158d91f83aef4b4d351",
    "realistic_vision_v3.0_q6p_q8p.ckpt":
      "6a4294760fb82295522cd7d610c95269070c403e5a4b41f67ce2db93fd93ee3a",
    "realistic_vision_v3.0_clip_vit_l14_f16.ckpt":
      "71f1c7726f842d72fe04a7e17ee468c32752d097b89e9114708c0dc13a0060a2",
    "dreamshaper_v6.31_q6p_q8p.ckpt":
      "14a9c0e4a5ebb4a66d4fd882135e60a3951e5d1d96e802cbf2106e91427e349f",
    "dreamshaper_v6.31_clip_vit_l14_f16.ckpt":
      "7384f31ea620891a7bca84c3b537beda7dbc5473873c90810b797e14ab263fc4",
    "open_clip_vit_bigg14_f16.ckpt":
      "1bc61283f12c3b923f4366a27d316742c0610aa934803481f0b5277124b9a8f4",
    "sd_xl_base_0.9_f16.ckpt": "e7613b7593f8f48b3799b3b9d2308ec2e4327cdd5f4904717d10846e0d13e661",
    "sd_xl_refiner_0.9_f16.ckpt":
      "b6e830f2d2084ca078178178aa67b31d85b17a772304e2ed39927e2f39487277",
    "sdxl_vae_f16.ckpt": "275decbdbe986f55bb20018bd636e3b0a8b0a6a3b8c28754262dcb84f33a62d7",
    "sdxl_vae_v1.0_f16.ckpt": "8ceb1b62fc9b88c20a171452fef55e3a5546cc621c943c78188f648351b4d7e4",
    "sd_xl_base_1.0_f16.ckpt": "741f813f9f7f17bf9e284885fa73b5098a30dc6db336179116e8749da10657a3",
    "sd_xl_refiner_1.0_f16.ckpt":
      "73abf6538793530fe3a2358a5789b7906db4e6dc30ce8d9d34b87a506fa2e34c",
    "sd_xl_base_1.0_q6p_q8p.ckpt":
      "796210c27eec08fd7ea01ad42eaf93efac5783b3f25df60906601a0a293a8f45",
    "sd_xl_refiner_1.0_q6p_q8p.ckpt":
      "be4f78ff34302d1cfbc91c1e83945e798bc58b0bc35ac08209d8d5a66b30c214",
    "ssd_1b_f16.ckpt": "8fed449f74cefadf9f10300eaa704d2fa0601bf087c1196560ce861aa6ab3d68",
    "ssd_1b_q6p_q8p.ckpt": "a4096821ac5fbc9c34be2fe86ca5b0e9d2f0cc64fd9c3ba47e1efe02cec5da09",
    "lcm_sd_xl_base_1.0_f16.ckpt":
      "937a0851d1c3fbb7b546d94edfad014db854721c596e0694d9e4ca7d6e8cd8de",
    "lcm_sd_xl_base_1.0_q6p_q8p.ckpt":
      "0830466d22f5304f415e2d96ab16244f21a2257d5e906ed63a467a393a38c250",
    "lcm_ssd_1b_f16.ckpt": "e1156cc6e6927a462102629d030e3d6377e373664201dad79fb1ff4928bb85b0",
    "lcm_ssd_1b_q6p_q8p.ckpt": "959d09951bdba0a73fafb6a69fed83b21012cbc4df78991463bbd84e283cc6fe",
    "sd_xl_turbo_f16.ckpt": "c85ea750f1ff5d17c032465c07f854eaf5f1551e27bd85dbe9c2d1025a41e004",
    "sd_xl_turbo_q6p_q8p.ckpt": "a8072ace4eb3d6590db8abe8fda6c0c22f4c3e68efb86f0e58a27dc4f68731ef",
    "open_clip_vit_h14_vision_model_f16.ckpt":
      "87b70da1898b05bfc7b53e3587fb5c0383fe4f48ccae286e8dde217bbee1e80d",
    "svd_i2v_1.0_f16.ckpt": "5751756a84bd9b6c91d2d6df7393d066d046e8ca939a8b8fa4ac358a07acaf94",
    "svd_i2v_1.0_q6p_q8p.ckpt": "5c8e4c1a1291456c5516c4c66d094eada0e11660c7b474cc39e45c9ceff27309",
    "svd_i2v_xt_1.0_f16.ckpt": "e5fd1a2f5fb7f1a13424e577a13c04dfd873b1cc6e3cdebc4c797d97d21a6865",
    "svd_i2v_xt_1.0_q6p_q8p.ckpt":
      "f3c4a06c1a1cb71a6b032e2ceb2d04e1d9c8457c455f8984f5324bbd8ba6d2e2",
    "fooocus_inpaint_sd_xl_v2.6_f16.ckpt":
      "f93886d787043cab976d31376b072bdc320185606331349ace9b48c41eeda867",
    "fooocus_inpaint_sd_xl_v2.6_q6p_q8p.ckpt":
      "f299e673da2d0da8ffccd6a01e9901261a9091a278032316c3218598ee9b5f2d",
    "svd_i2v_xt_1.1_f16.ckpt": "cd4d0c43c6cd3a3af51e35d465e2cec5292778f9cd12c92b64873f59de6ef314",
    "svd_i2v_xt_1.1_q6p_q8p.ckpt":
      "61c6fe0cce4d91fc1b83dd65f956624dc2c996fb21fdc4fa847fbf4bc97e0030",
    "wurstchen_3.0_stage_a_hq_f16.ckpt":
      "ad9d2b43ceb68f9bb9d269a6a5fd345a5f177a0f695189be82219cb4d2740277",
    "wurstchen_3.0_stage_b_q6p_q8p.ckpt":
      "b0611225cf2f2a7b9109ae18eaf12bfe04ae60010ac5ea715440d79708e578b8",
    "wurstchen_3.0_stage_c_f32_q6p_q8p.ckpt":
      "0e57d6f6c7749a34ea362a115558aeeb209da82e54b06e3b97433ed64b244439",
    "wurstchen_3.0_stage_b_f16.ckpt":
      "a541358038cb86064a4d43bd0b6dab1cb95129520fca67eb178bce3baccc1d02",
    "wurstchen_3.0_stage_c_f32_f16.ckpt":
      "aa05651d1920d1fd0b70d06397548bf9e77fac93ff4b4bc9bc98cea749e5a8db",
    "playground_v2.5_f16.ckpt": "9a8e167526a65d5caebfd6d5163705672cfd4d201cb273d11c174e46af041b4a",
    "playground_v2.5_q6p_q8p.ckpt":
      "18ddd151c7ae188b6a0036c72bf8b7cd395479472400a3ed4d1eb8e5e65b36e3",
    "open_clip_vit_h14_visual_proj_f16.ckpt":
      "ef03b8ac7805d5a862db048c452c4dbbd295bd95fed0bf5dae50a6e98815d30f",
    "wurstchen_3.0_stage_c_effnet_previewer_f32_f16.ckpt":
      "fd1c698895afc14e68a0d7690c787400796114d2bfac0df254598d8207f93f0f",
    "t5_xxl_encoder_q6p.ckpt":
      "8f8fa0acc618df6f225122b3d03b6f60034490d9f9fd3b8799e7faa3e08943b7",
    "sd3_vae_f16.ckpt":
      "51d42d4745456396f0cbb034f4eb9d6495cc2a552ca4af7ba80d64fdba5f9678",
    "sd3_medium_q8p.ckpt":
      "a313371538d8018ee6f3f3b6aa3e08bff64bfa3a56c1f777b90973f71138b3a2",
    "sd3_medium_f16.ckpt":
      "9ee38fee52867678c21afffd7c176443a61e30eed728c1a28e2ff4982fe89bee",
    "pixart_sigma_xl_2_1024_ms_f16.ckpt":
      "b78f0f8d4988b6edf38eeff8c1d33d2b4ffca1fa79c4b45f51b8647aa3b625a0",
    "pixart_sigma_xl_2_1024_ms_q8p.ckpt":
      "d5379d9f7ad18e3dd8b6f4b564df1f03bc1e8377e1486bb793467fc4fab6ae5c",
    "pile_t5_xl_encoder_q8p.ckpt":
      "ef8b228e915bb21101c4c34e89039e2c42ddba843dae4b1e4f813a4785b1df1b",
    "auraflow_v0.1_q8p.ckpt": "30ebb3796987ff2f79cb67b16e72f4ba5e31dd4706af0e8d0d91fb16165c71ee",
    "auraflow_v0.1_f16.ckpt": "8c5e7ba677ccd11f899f2fb4092ab7cc4ad7686d01be172497e55eeba01c5bb0",
    "auraflow_v0.1_q5p.ckpt": "9aab1942ea2f025846d5d1dcd2ae5df762c1a10887c807b603c223ba8e5e5ad7",
    "auraflow_v0.2_q8p.ckpt": "cda840bce05ada4c97d95080160f18dc594b6c5f2d4da45c33db51c37c070170",
    "auraflow_v0.2_f16.ckpt": "727622af19710b8014da024c7294573c02fefb7be83e178fa4c2b50a9d2bc922",
    "auraflow_v0.2_q5p.ckpt": "b3d4a2c3be69e285028de0d61a17a5fbe34b9e1504725f32f52e75b8d9d8a2cc",
    "flux_1_vae_f16.ckpt": "453d09645419d1ffc2f641e4a4c6ccf75f69c7215938a285e474ece3762fe293",
    "flux_1_schnell_q8p.ckpt": "26a38212290a928aad21d4d9a6e534cca6c06ddb7ce0a926ac31533500e39f64",
    "flux_1_schnell_f16.ckpt": "6fad328261de43847bf6a53a075445e90c5fd90f65c4a68dc538fcb7aa5f13a2",
    "flux_1_schnell_q5p.ckpt": "37a28dcba93e23e4433b64d621b5352f4651d46eaf40251351d8a553642b907b",
    "wan_v2.1_1.3b_480p_f16.ckpt":
      "d8f0e77085890f86b2d61095d1340bb94bbf876edac5a472693e7f047e5ac4a6",
    "wan_v2.1_1.3b_480p_q8p.ckpt":
      "2aa588d0c001d190e8a3a6b176835e89c97800479b8fe5d1837fe4514f9bf805",
    "umt5_xxl_encoder_q8p.ckpt": "72ef62d22c09a3b764ac9e6fe0100f4029619fb3ff8ccec3432e509487b29831",
    "wan_v2.1_video_vae_f16.ckpt":
      "4c518b128b3c1f2ea164aa46269d8875b4be3661d1fc0fba2709d03fa94e418b",
    "wan_v2.1_14b_720p_q8p.ckpt":
      "b21d70e196e5dfd4c3238607c9c3a13150d4aae04848245ed57241b83ee586bd",
    "wan_v2.1_14b_720p_q6p_svd.ckpt":
      "bc421931cd177c25d419123ca5f569b45d6942716e867520481a97f6bb988896",
    "open_clip_xlm_roberta_large_vit_h14_f16.ckpt":
      "362c9940a36acce5a4e13b9167d5daebd005ac026443cd37b7955ac0acd72083",
    "wan_v2.1_14b_i2v_480p_q8p.ckpt":
      "ad9ba7c4db022abd89e9d9f02061f65335b6c222b075cb67831ea5118ecf480a",
    "wan_v2.1_14b_i2v_720p_q8p.ckpt":
      "eacd5ab2d91f982e68c5a786735a88f504c303e34907604c89a01f2083654a6d",
    "wan_v2.1_14b_i2v_480p_q6p_svd.ckpt":
      "dba1a4fe5c29eb33479759b00ed309c32cc63d92663c7bd74c3d0aedd2dbd0b9",
    "wan_v2.1_14b_i2v_720p_q6p_svd.ckpt":
      "16bc54134e4e16998df12713722d8cd1038f2cdd0955023835d2801bae720c54",
    "long_clip_vit_l14_f16.ckpt":
      "82031eaa248d543a072af378ccd6280cd3be1d07f8733c5d15f9ec4feb82501a",
    "long_open_clip_vit_bigg14_f16.ckpt":
      "6beca0db6c1f84b84b6facb0c2ce4abe56fb220be978ee1438064797861f949b",
    "llama_3.1_8b_instruct_q8p.ckpt":
      "9b0a80a78041ea4ad3c608f7255ec2186afb3ce5d504f955cfd821afc590da57",
    "hidream_i1_fast_q8p.ckpt": "a5f17f1a86a903b8ce8a4fe147ddd82d3c166842b5c67116eb7bf4692c22b2e8",
    "hidream_i1_dev_q8p.ckpt": "1ff76a095b8f75e3047409e1704d1fbbb6c923853a5c59cc699a7b94a5b2c83e",
    "hidream_i1_full_q8p.ckpt": "24c76c58d296f467e458a5bec4edd512ddf697ceb6739239e8a2494e5a50cf4e",
    "hidream_i1_fast_q5p.ckpt": "ce7254c72257edd78b821881c72e9a91afe2752a187968fc4eb3a1648ec35053",
    "hidream_i1_dev_q5p.ckpt": "9779d730f8f2258cdf721a770c91ade978836f11510d9173075f89bc8f8be3e3",
    "hidream_i1_full_q5p.ckpt": "a1f5371896c93c7fb55328331c226eb9bcaabf804a142f6c85f5684d5bd4c3ae",
    "hidream_e1_full_q8p.ckpt": "0f24a6f94cd7105a1bfea195e4ce3e064427833e97390fc1ce6bc6945c8cb93f",
    "hidream_e1_full_q5p.ckpt": "63ff5d43c474937b3f83fe9220a2dbbea7d215fc0585db0fcf12e37ac87dc60c",
    "hidream_e1_1_q8p.ckpt": "b1fe02e5992e6696947f76d5afe0ea1aef7908f9947ae2d665b1d21661aeaf46",
    "hidream_e1_1_q5p.ckpt": "c50ddbdccc8e59ea03f1d382dbdb79c3ad9ad1ec8f543173241469324a4396dd",
    "wan_v2.2_a14b_hne_t2v_q8p.ckpt":
      "63b3dc2eaac38d1019017c4745ea763e9894b5aa55a97fb52dfe26985abc42e6",
    "wan_v2.2_a14b_hne_t2v_q6p_svd.ckpt":
      "f3d3adb772520896bc5d016d00f9159c8595fbdec2ecc33dff2572bffb39cadd",
    "wan_v2.2_a14b_lne_t2v_q8p.ckpt":
      "5fe77d9998141a667071353658d71acef656389210327391c832d39b4ee39671",
    "wan_v2.2_a14b_lne_t2v_q6p_svd.ckpt":
      "861aef3cb11c3dd44052e79a5cbd4740550dc05a7d119cf3069de152958c9e1e",
    "wan_v2.2_a14b_hne_i2v_q8p.ckpt":
      "bf44ed82e723cf3f2469c9a89fd5a98296c924d489b883e7dadef4c122ad295d",
    "wan_v2.2_a14b_hne_i2v_q6p_svd.ckpt":
      "565f535dc5264a9264ec17e2e8396d9a23509be92d71a4930c40fafa16b29685",
    "wan_v2.2_a14b_lne_i2v_q8p.ckpt":
      "51486a592ba0190c333ba4e071aa9ed2b502c8aa5b79cb5d38325e399ac6c129",
    "wan_v2.2_a14b_lne_i2v_q6p_svd.ckpt":
      "ed0c61db1dccce4beaa2a805792de43c77c9b87d876e3633ef60e4cb716e153a",
    "qwen_2.5_vl_7b_q8p.ckpt": "513b759b24619946d3ca13c0bf57464a32098593bbf8342001358fbfa51f78b1",
    "qwen_image_vae_f16.ckpt": "701e7c46ed6c2fa8036543780317e06d264374f9b4dbfc22f27c0b3181bb988a",
    "qwen_image_1.0_q8p.ckpt": "34e25c219945f5887bbab5ed0be39db78b8635f4986f6715b80eda2a3d581081",
    "qwen_image_1.0_q6p.ckpt": "51bf057484c66cb7c4e6e3bab80b99e99aa20af043f42086ce5080dc01986b62",
    "qwen_2.5_vl_7b_vit_f16.ckpt":
      "e670fe5e19d02d3bce73bb655824855869ad135a21ba88170966f13700a60504",
    "qwen_image_edit_1.0_q8p.ckpt":
      "3e79e7b19b2ca152c7a13feeec4ba8545d383d9375a8f9072db10d364b484d6d",
    "qwen_image_edit_1.0_q6p.ckpt":
      "c816944982f8fba4ae0bcf36279f25564825386416f03088c4f223c9911407c5",
    "wan_v2.2_5b_ti2v_f16.ckpt": "58b05be04f2f66f4d9e0c7605aebf9eb82c80b0c2c9498a43a1b95e9600f3389",
    "wan_v2.2_5b_ti2v_q8p.ckpt": "e01bc2a772e1a4c5bb577fb2a98a9070638c359ecaf2ac62e6a017f517b9ab9b",
    "wan_v2.2_video_vae_f16.ckpt":
      "5e4dc4638ff09ed3a0173f907a78d2dde90c56ef855070059b594dd0c7a75a1e",
    "qwen_image_edit_2509_q8p.ckpt":
      "d3e5bf87e24d732285c39e76efcb4fadcbac0f37204f99e27ba6c7083f116a2a",
    "qwen_image_edit_2509_q6p.ckpt":
      "3ab7456121e770008e6f434405548a6ca5b3b83053b8acee89947939252146f5",
    "qwen_image_1.0_bf16_q8p.ckpt":
      "4c38f4da338ec63038adea4cc076e7a46dcd70c79086e7a7b53b7d880a80fcd9",
    "qwen_image_1.0_bf16_q6p.ckpt":
      "52aba4938362a0faad933bb91e20716e3a41c196b799daadf411c9689d3772a7",
    "qwen_image_edit_2509_bf16_q8p.ckpt":
      "01a56e3f4c5d2d1aae0e8d8cee6405498679722128cd47c2c93d6323e24d7552",
    "qwen_image_edit_2509_bf16_q6p.ckpt":
      "5ace7c50b133854b8242a2b03ab9b687036c126f2c3f0d80571cd17e977c3eaa",
    "qwen_3_vl_4b_instruct_q8p.ckpt":
      "81bb4dfdeaa101a6948174ca827f7ebec0c8864f1f878f91231a7446ce96a9e2",
    "z_image_turbo_1.0_q6p.ckpt":
      "252683bda4661be9c64d94e718c2a9c36a6217867464ca09d03db7013a45f20f",
    "z_image_turbo_1.0_q8p.ckpt":
      "fb2f636c4f310c092fd87b6c07aabb1f4549fd4d74b66b09a5d034c95da655e6",
    "qwen_image_layered_1.0_bf16_q6p.ckpt":
      "28c5a5a1417d955ef802f318e489fedefc2ed4c39d79032ea6cdc84b7eafc894",
    "qwen_image_layered_1.0_bf16_q8p.ckpt":
      "de2571a4cbb59360408a2eec5ec6fee0d19998a0355bb94f362cd1bb43428c1b",
    "qwen_image_layered_vae_f16.ckpt":
      "52913c47f68c3aaa371b6819608f1a0d411d9c269d3d38552d306f6957fe482b",
    "qwen_image_edit_2511_bf16_q6p.ckpt":
      "27f9701bed3148168c5a0e8038c26e4e9c589ca9095e072479af9e843abfd680",
    "qwen_image_edit_2511_bf16_q8p.ckpt":
      "e7f0ab1da03ca974af35b397b8406e3b88e81ddf456cf7648d734e84b3595959",
    "qwen_image_edit_2511_q6p.ckpt":
      "05af17e79ab30b17cbfe3645acc3c627fcf5d05b2622f8f854b9f19c02f12e44",
    "qwen_image_edit_2511_q8p.ckpt":
      "efe1adbb4e6c24a2e6ca69ec0200c80a0490002b0e9dcb9faf1b7ca8e8b7e391",
    "qwen_image_2512_bf16_q6p.ckpt":
      "885933321537a509e1a200d5ccddecbeaa4d0b2ef72886452dabf2425ddb026a",
    "qwen_image_2512_bf16_q8p.ckpt":
      "adeadfb2217c1087a4d91b8dde8557f922ceb41ebb774f69c920a87036b99e96",
    "qwen_image_2512_q6p.ckpt":
      "b6478e7e8193154778661f3cdb77d791725349ad8f1bded212109833dc7c569d",
    "qwen_image_2512_q8p.ckpt":
      "17f2bab8c39a96cca1b4dc99b8b37dfe7fda779c3e95a6cae24272d1e08ab7e5",
    "qwen_3_4b_q8p.ckpt": "a24e3f832917aafe7f6186e6dfb96ab19a4a0729cdf60a875b5da20024333350",
    "flux_2_klein_4b_q8p.ckpt": "430ba0f94ee0851a7f95c1383527f931afa6219898a51c5aa8e37979e9e4c86a",
    "flux_2_klein_4b_q6p.ckpt": "ba42005a6d4e6c48430787db98ae8913aced4654682b59d0d0f1d9ccf820fe30",
    "ltx_2_19b_dev_q8p.ckpt": "421fb98cc3910463d42a8dd6f96e8821a2c102d347e4ccfcbc1f7d7d2c7a1f4d",
    "ltx_2_19b_distilled_q8p.ckpt":
      "33be843243e29e776d64f730d07c1d22cb1f17727b844e27d5ee14004c109ed6",
    "ltx_2_19b_dev_q6p.ckpt": "a1bbe2cb2282daf98bdb7a879f5894e9212cf11bd00fbab819d7cd76aeacff0c",
    "ltx_2_19b_distilled_q6p.ckpt":
      "e525fa584673cc413274f4bd4b23d9b8b7cf0e88f445d547f87224dc8c16f298",
    "gemma_3_12b_it_qat_q8p.ckpt":
      "9292a7107d30b6eef8282654c9746ac2e868a6d00704bdb9cfdadf45d4c325af",
    "ltx_2_audio_video_vae_f16.ckpt":
      "7e40bdffaf9f67227c6be38ee16e308dabf896545593cecec83576d9c81aa35e",
    "z_image_1.0_q6p.ckpt":
      "74dba3b854b61f422fa40851c046aa69c6f9b2659a671ae3938062cdfb235d70",
    "z_image_1.0_q8p.ckpt":
      "4233ac5ce386e824fce8628070ee985e070de55925c63bb0d6bf7890d523631e",
    "ltx_2.3_22b_distilled_q6p.ckpt":
      "66f952d338213a0f472a25bcc8fc38fc4176ce3080683891c72ae2f40dcdcff9",
    "ltx_2.3_22b_distilled_q8p.ckpt":
      "5736128a9cea4a3ff1f1dca6b991936920916a248b7c96aa8a8cd831e294c21a",
    "ltx_2.3_audio_video_vae_f16.ckpt":
      "72e4c23b34e631599773b05beb7e355c00d36feb983fe9746d351b58a4be4f02",
    "ltx_2.3_22b_dev_q6p.ckpt":
      "676fc93063045e107dc642563c475e25306969b0791ab933eaeba634d673fe04",
    "ltx_2.3_22b_dev_q8p.ckpt":
      "dcf1fe7b365da89ef9bcacaf488682addbef2d255a31d7f0a580d5c6841d6e2a",
    "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt":
      "249f5bd521472a952775e0cb9729836d2fc9ab488ca64e08329e19dd07060157",
    "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt":
      "30f695c0b24a59a6490f1552c025124d713f61d4e6a90da6b0def038c8caa845",
    "ltx_2.3_22b_distilled_i8x.ckpt":
      "7da4b90696da134e02024df1501b1cb7f352b4714c4ccb0223fbf88b101829dc",
    "ltx_2.3_22b_dev_i8x.ckpt":
      "0f6444dd1784417618ccb7648416df931c572913a4840c689a48ed272a6d2679",
    "z_image_turbo_1.0_i8x.ckpt":
      "1f12425e76cc525de20d8047aac681949d964c986c59dca8615b17ce7601ce29",
    "z_image_1.0_i8x.ckpt":
      "ae7436f413ebd81250983005babb5bfdb562b0c97046ec582ade04cffed3c315",
    "flux_2_klein_4b_i8x.ckpt":
      "19985019d78456d6de025a27f048ddc6aefd4e2e28e4ea1827126841932bb645",
    "flux_2_klein_base_4b_i8x.ckpt":
      "e0ebcfa55e32ed9cc5d3c1c679fae1ecd5e518c00af9411afcbb5da1c5b36472",
    "qwen_image_2512_i8x.ckpt":
      "86746c3c385255b1b71a40ab8c6b4b758c14eb27dad64f4ae1742521294cf868",
    "qwen_image_2512_bf16_i8x.ckpt":
      "9bcedf9b533a7d6a734faa15801085cf4c74678119dc648851352e16b8fd80aa",
    "qwen_image_edit_2511_i8x.ckpt":
      "12bb11eb40d79935b39cbc3899a51876786d50ac4afd1b09163cf91cd67a7a8b",
    "qwen_image_edit_2511_bf16_i8x.ckpt":
      "f4430119fb5ecf2ede5b9bdfe42d57e79668b73eba79137588e74a763aca17a4",
    "qwen_image_layered_1.0_bf16_i8x.ckpt":
      "6bbe85a146a49f485341a952396245bb2fb758671ce794eb6b4cedadddb9b87f",
    "wan_v2.2_a14b_hne_i2v_i8x.ckpt":
      "f5bb8c2e570b1a635517377365260821836515960bc2e1361cfb1275dc577c26",
    "wan_v2.2_a14b_hne_t2v_i8x.ckpt":
      "4e413a122e21658d287db3b40343c6956e4f4154e25b7c6b40a08d9c03170af5",
    "wan_v2.2_a14b_lne_i2v_i8x.ckpt":
      "72d73d00682f26cd1cef72f0377b261891e5dcfb1b1fbc84c99afa2790d569ec",
    "wan_v2.2_a14b_lne_t2v_i8x.ckpt":
      "233cac703a303709728a535f228166c10614a4ff55e4eda8196274b5787b4b55",
    "hidream_i1_dev_i8x.ckpt":
      "5bdc535a11b1890616011d7fd1648758c1b5ed0be179b7cd361369ebd39aa1ba",
    "hidream_i1_fast_i8x.ckpt":
      "ad7fb5cb2fb935bfc33cb884868ff2319521e3f1ab375d9b9dad288fd0c73788",
    "hidream_i1_full_i8x.ckpt":
      "cf53865d9015ac08fd9e64c990160cf191b6336375bef169e669abcb5d861431",
    "ernie_image_i8x.ckpt":
      "2ae4d881a9d740f4aa18ce270563df105841111ff466505b35054801085b8c5a",
    "ernie_image_q8p.ckpt":
      "724f86513cca83a551aff17877851819ad4da7dfdbbc873996cbfda31c0e3323",
    "ernie_image_q6p.ckpt":
      "63a2b89afe233a3f0108b6b9e0e8dd73d429a11aa03d2abbfc8abb06e3e932bd",
    "ernie_image_turbo_i8x.ckpt":
      "6286758f178bee9dbdcfadb5c9781372b6222e2c546aa0b223bec81cb80d34de",
    "ernie_image_turbo_q8p.ckpt":
      "939315c139f2a47a307bb9405a6992325fc416cfaf60de36e9537be8e8b5ca85",
    "ernie_image_turbo_q6p.ckpt":
      "592ed1c171f0efcfb3633a948c98326cd66d0d1dd3f9fce05b9042be0c8be861",
    "ministral_3_3b_q8p.ckpt":
      "64be13a9f12230731be6acf01df1aecced00b7ffe6d5d9cc552321edfe3a46bd",
    "ltx_2.3_22b_distilled_1.1_q8p.ckpt":
      "ebb0c337ef43a39d6952ca35a1f245b33bfc3e4c9d9451e7138b926075f0b8dd",
    "ltx_2.3_22b_distilled_1.1_q6p.ckpt":
      "f7c804dd0e4ae14af2ff200fc96b2848b2967f7595bac830bf270ea39b47a611",
    "ltx_2.3_22b_distilled_1.1_i8x.ckpt":
      "c2bb0ac129aab08a0759b1c4d9a3e0c2202a4ae7a4e942c32d362f6559613d92",
    "seedvr2_vae_f16.ckpt":
      "af2a4a8eeae04372667ba4d6f6dd1337af0b3685d0e335dfc5abf71374a24234",
    "seedvr2_3b_q8p.ckpt":
      "cdde46771a8af490b97853c8fb255864a64d1528b40b1ebf2dee161db2b76fcd",
    "seedvr2_3b_i8x.ckpt":
      "c6cc6f936f3acea3bd3809f176b27696aee3f7a995cbb9083b45b03a3233698e",
    "seedvr2_3b_q6p.ckpt":
      "160a5daefec6292b50e41a8dd74e0c91702f5571f4f5e43f51c58eb01718e0ea",
    "seedvr2_7b_q8p.ckpt":
      "acb88f9c581d2668eead19742c4fcc8f7af515ff3abf9a4ffd7450e731bed9ce",
    "seedvr2_7b_i8x.ckpt":
      "05daa3433a3719854498ea3694a14e3ecde2c1a8f2a4fc7c4f02ac9ff7ba5a78",
    "seedvr2_7b_q6p.ckpt":
      "92d193e4da2b2bea6e7877ca6c011d693da6dbc9bc9b735aee8c102905fa446f",
    "hidream_o1_dev_2604_i8x.ckpt":
      "d8588595c6177ed451d7843af999577f424376c4d2ef77f9408883a13ec0b231",
    "krea_2_turbo_i8x.ckpt":
      "e808501ada16a6dedc5b6d8b6426243d4b89d3be1ac9a02706e855684120e87f",
    "krea_2_raw_i8x.ckpt":
      "e8ab2f70261904efbff9fc04f10dcd64043194db1cf5a7e2c5ff48f7cc2ba80f",
    "qwen_3_vl_4b_q8p.ckpt":
      "be71703dbe695d2aee038539ac0a4dcba02a3ffdc4df4e5fe0b2c3f70a47c151",
    "ideogram_4_i8x.ckpt":
      "c82dd45d6fba7ad038e8f148760ed5cf65fe4e53c56b57b778fee26275e73503",
    "qwen_3_vl_8b_instruct_q8p.ckpt":
      "54d29b3e951c233ae800604848b83422a48629857c70fb83c4620eb31d900b9a",
  ]

  public static let defaultSpecification: Specification = builtinSpecifications[0]

  public static let builtinSpecifications: [Specification] = [
    Specification(
      name: "LTX-2.3 22B [distilled] 1.1", file: "ltx_2.3_22b_distilled_1.1_q8p.ckpt", prefix: "",
      version: .ltx2_3, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2.3_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2.3_22b_distilled_1.1_q8p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      latentsUpscalers: [
        .init(file: "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt", scale: .x2),
        .init(file: "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt", scale: .x1_5),
      ],
      note:
        "[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a significant update to LTX-2 with improved audio and visual quality and enhanced prompt adherence. It is a state-of-the-art open-source audio-video foundation model developed by Lightricks and can generate synchronized video and audio within a single model. The [distilled] checkpoint is the distilled version of the full model and is optimized for fast inference; for best results, use 8 sampling steps and set Text Guidance to 1.0.",
      copyright: "© 2026 Lightricks", huggingFaceLink: "Lightricks/LTX-2.3"
    ),
    Specification(
      name: "LTX-2.3 22B [distilled] 1.1 (8-bit S)", file: "ltx_2.3_22b_distilled_1.1_i8x.ckpt",
      prefix: "",
      version: .ltx2_3, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2.3_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2.3_22b_distilled_1.1_i8x.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      latentsUpscalers: [
        .init(file: "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt", scale: .x2),
        .init(file: "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt", scale: .x1_5),
      ],
      note:
        "[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a significant update to LTX-2 with improved audio and visual quality and enhanced prompt adherence. It is a state-of-the-art open-source audio-video foundation model developed by Lightricks and can generate synchronized video and audio within a single model. The [distilled] checkpoint is the distilled version of the full model and is optimized for fast inference; for best results, use 8 sampling steps and set Text Guidance to 1.0.",
      copyright: "© 2026 Lightricks", huggingFaceLink: "Lightricks/LTX-2.3"
    ),
    Specification(
      name: "LTX-2.3 22B [distilled] 1.1 (6-bit)", file: "ltx_2.3_22b_distilled_1.1_q6p.ckpt",
      prefix: "",
      version: .ltx2_3, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2.3_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2.3_22b_distilled_1.1_q6p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      latentsUpscalers: [
        .init(file: "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt", scale: .x2),
        .init(file: "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt", scale: .x1_5),
      ],
      note:
        "[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a significant update to LTX-2 with improved audio and visual quality and enhanced prompt adherence. It is a state-of-the-art open-source audio-video foundation model developed by Lightricks and can generate synchronized video and audio within a single model. The [distilled] checkpoint is the distilled version of the full model and is optimized for fast inference; for best results, use 8 sampling steps and set Text Guidance to 1.0.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "LTX-2.3 22B [distilled]", file: "ltx_2.3_22b_distilled_q8p.ckpt", prefix: "",
      version: .ltx2_3, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2.3_audio_video_vae_f16.ckpt", modifier: .kontext, deprecated: true,
      clipEncoder: "ltx_2.3_22b_distilled_q8p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      latentsUpscalers: [
        .init(file: "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt", scale: .x2),
        .init(file: "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt", scale: .x1_5),
      ],
      note:
        "[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a significant update to LTX-2 with improved audio and visual quality and enhanced prompt adherence. It is a state-of-the-art open-source audio-video foundation model developed by Lightricks and can generate synchronized video and audio within a single model. The [distilled] checkpoint is the distilled version of the full model and is optimized for fast inference; for best results, use 8 sampling steps and set Text Guidance to 1.0.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "LTX-2.3 22B [distilled] (8-bit S)", file: "ltx_2.3_22b_distilled_i8x.ckpt", prefix: "",
      version: .ltx2_3, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2.3_audio_video_vae_f16.ckpt", modifier: .kontext, deprecated: true,
      clipEncoder: "ltx_2.3_22b_distilled_i8x.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      latentsUpscalers: [
        .init(file: "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt", scale: .x2),
        .init(file: "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt", scale: .x1_5),
      ],
      note:
        "[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a significant update to LTX-2 with improved audio and visual quality and enhanced prompt adherence. It is a state-of-the-art open-source audio-video foundation model developed by Lightricks and can generate synchronized video and audio within a single model. The [distilled] checkpoint is the distilled version of the full model and is optimized for fast inference; for best results, use 8 sampling steps and set Text Guidance to 1.0.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "LTX-2.3 22B [distilled] (6-bit)", file: "ltx_2.3_22b_distilled_q6p.ckpt", prefix: "",
      version: .ltx2_3, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2.3_audio_video_vae_f16.ckpt", modifier: .kontext, deprecated: true,
      clipEncoder: "ltx_2.3_22b_distilled_q6p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      latentsUpscalers: [
        .init(file: "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt", scale: .x2),
        .init(file: "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt", scale: .x1_5),
      ],
      note:
        "[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a significant update to LTX-2 with improved audio and visual quality and enhanced prompt adherence. It is a state-of-the-art open-source audio-video foundation model developed by Lightricks and can generate synchronized video and audio within a single model. The [distilled] checkpoint is the distilled version of the full model and is optimized for fast inference; for best results, use 8 sampling steps and set Text Guidance to 1.0.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "LTX-2.3 22B [dev]", file: "ltx_2.3_22b_dev_q8p.ckpt", prefix: "",
      version: .ltx2_3, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2.3_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2.3_22b_dev_q8p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      latentsUpscalers: [
        .init(file: "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt", scale: .x2),
        .init(file: "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt", scale: .x1_5),
      ],
      note:
        "[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a significant update to LTX-2 with improved audio and visual quality and enhanced prompt adherence. It is a state-of-the-art open-source audio-video foundation model developed by Lightricks and can generate synchronized video and audio within a single model. The [dev] checkpoint is the full model variant and is fully trainable. For best results, set Text Guidance to 3.5 and use 20–30 sampling steps.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "LTX-2.3 22B [dev] (8-bit S)", file: "ltx_2.3_22b_dev_i8x.ckpt", prefix: "",
      version: .ltx2_3, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2.3_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2.3_22b_dev_i8x.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      latentsUpscalers: [
        .init(file: "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt", scale: .x2),
        .init(file: "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt", scale: .x1_5),
      ],
      note:
        "[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a significant update to LTX-2 with improved audio and visual quality and enhanced prompt adherence. It is a state-of-the-art open-source audio-video foundation model developed by Lightricks and can generate synchronized video and audio within a single model. The [dev] checkpoint is the full model variant and is fully trainable. For best results, set Text Guidance to 3.5 and use 20–30 sampling steps.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "LTX-2.3 22B [dev] (6-bit)", file: "ltx_2.3_22b_dev_q6p.ckpt", prefix: "",
      version: .ltx2_3, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2.3_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2.3_22b_dev_q6p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      latentsUpscalers: [
        .init(file: "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt", scale: .x2),
        .init(file: "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt", scale: .x1_5),
      ],
      note:
        "[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a significant update to LTX-2 with improved audio and visual quality and enhanced prompt adherence. It is a state-of-the-art open-source audio-video foundation model developed by Lightricks and can generate synchronized video and audio within a single model. The [dev] checkpoint is the full model variant and is fully trainable. For best results, set Text Guidance to 3.5 and use 20–30 sampling steps.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "LTX-2 19B [distilled]", file: "ltx_2_19b_distilled_q8p.ckpt", prefix: "",
      version: .ltx2, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2_19b_distilled_q8p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[LTX-2](https://huggingface.co/Lightricks/LTX-2) is a state-of-the-art open-source audio-video foundation model developed by Lightricks. It can generate synchronized video and audio within a single model. The [distilled] checkpoint is optimized for fast inference; for best results, use 8 sampling steps and set Text Guidance to 1.0.",
      copyright: "© 2026 Lightricks", huggingFaceLink: "Lightricks/LTX-2"
    ),
    Specification(
      name: "LTX-2 19B [distilled] (6-bit)", file: "ltx_2_19b_distilled_q6p.ckpt", prefix: "",
      version: .ltx2, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2_19b_distilled_q6p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[LTX-2](https://huggingface.co/Lightricks/LTX-2) is a state-of-the-art open-source audio-video foundation model developed by Lightricks. It can generate synchronized video and audio within a single model. The [distilled] checkpoint is optimized for fast inference; for best results, use 8 sampling steps and set Text Guidance to 1.0.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "LTX-2 19B [dev]", file: "ltx_2_19b_dev_q8p.ckpt", prefix: "",
      version: .ltx2, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2_19b_dev_q8p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[LTX-2](https://huggingface.co/Lightricks/LTX-2) is a state-of-the-art open-source audio-video foundation model developed by Lightricks. It can generate synchronized video and audio within a single model. The [dev] checkpoint is the full model variant. For best results, set Text Guidance to 3.5 and use 20–30 sampling steps.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "LTX-2 19B [dev] (6-bit)", file: "ltx_2_19b_dev_q6p.ckpt", prefix: "",
      version: .ltx2, defaultScale: 12, textEncoder: "gemma_3_12b_it_qat_q8p.ckpt",
      autoencoder: "ltx_2_audio_video_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "ltx_2_19b_dev_q6p.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[LTX-2](https://huggingface.co/Lightricks/LTX-2) is a state-of-the-art open-source audio-video foundation model developed by Lightricks. It can generate synchronized video and audio within a single model. The [dev] checkpoint is the full model variant. For best results, set Text Guidance to 3.5 and use 20–30 sampling steps.",
      copyright: "© 2026 Lightricks"
    ),
    Specification(
      name: "ERNIE Image Turbo 1.0", file: "ernie_image_turbo_q8p.ckpt", prefix: "",
      version: .ernieImage, defaultScale: 16, textEncoder: "ministral_3_3b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", hiresFixScale: 24,
      note:
        "[ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) is Baidu's Apache 2.0-licensed distilled release of ERNIE-Image, optimized with DMD and RL for fast generation. It keeps the same focus on instruction following, text rendering, and structured layouts, while targeting strong results with CFG 1 and 8 sampling steps.",
      copyright: "© 2026 Baidu", huggingFaceLink: "baidu/ERNIE-Image-Turbo"
    ),
    Specification(
      name: "ERNIE Image Turbo 1.0 (8-bit S)", file: "ernie_image_turbo_i8x.ckpt", prefix: "",
      version: .ernieImage, defaultScale: 16, textEncoder: "ministral_3_3b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", hiresFixScale: 24,
      note:
        "[ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) is Baidu's Apache 2.0-licensed distilled release of ERNIE-Image, optimized with DMD and RL for fast generation. It keeps the same focus on instruction following, text rendering, and structured layouts, while targeting strong results with CFG 1 and 8 sampling steps.",
      copyright: "© 2026 Baidu", huggingFaceLink: "baidu/ERNIE-Image-Turbo"
    ),
    Specification(
      name: "ERNIE Image Turbo 1.0 (6-bit)", file: "ernie_image_turbo_q6p.ckpt", prefix: "",
      version: .ernieImage, defaultScale: 16, textEncoder: "ministral_3_3b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", hiresFixScale: 24,
      note:
        "[ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) is Baidu's Apache 2.0-licensed distilled release of ERNIE-Image, optimized with DMD and RL for fast generation. It keeps the same focus on instruction following, text rendering, and structured layouts, while targeting strong results with CFG 1 and 8 sampling steps.",
      copyright: "© 2026 Baidu"
    ),
    Specification(
      name: "ERNIE Image Base 1.0", file: "ernie_image_q8p.ckpt", prefix: "",
      version: .ernieImage, defaultScale: 16, textEncoder: "ministral_3_3b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", hiresFixScale: 24,
      note:
        "[ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) is Baidu's Apache 2.0-licensed 8B single-stream DiT text-to-image model. It focuses on instruction following, text rendering, and structured layouts such as posters, comics, and multi-panel compositions. The base model is intended for higher-fidelity generation, with 50 sampling steps recommended.",
      copyright: "© 2026 Baidu", huggingFaceLink: "baidu/ERNIE-Image"
    ),
    Specification(
      name: "ERNIE Image Base 1.0 (8-bit S)", file: "ernie_image_i8x.ckpt", prefix: "",
      version: .ernieImage, defaultScale: 16, textEncoder: "ministral_3_3b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", hiresFixScale: 24,
      note:
        "[ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) is a distilled rectified-flow image model from Baidu. It uses the Ministral 3 text encoder family together with the FLUX.2 VAE path, and works best with trailing samplers, CFG 1, and around 8 sampling steps.",
      copyright: "© 2026 Baidu", huggingFaceLink: "baidu/ERNIE-Image"
    ),
    Specification(
      name: "ERNIE Image Base 1.0 (6-bit)", file: "ernie_image_q6p.ckpt", prefix: "",
      version: .ernieImage, defaultScale: 16, textEncoder: "ministral_3_3b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", hiresFixScale: 24,
      note:
        "[ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) is Baidu's Apache 2.0-licensed 8B single-stream DiT text-to-image model. It focuses on instruction following, text rendering, and structured layouts such as posters, comics, and multi-panel compositions. The base model is intended for higher-fidelity generation, with 50 sampling steps recommended.",
      copyright: "© 2026 Baidu"
    ),
    Specification(
      name: "SeedVR2 3B", file: "seedvr2_3b_q8p.ckpt", prefix: "",
      version: .seedvr2_3b, defaultScale: 24, textEncoder: "seedvr2_3b_q8p.ckpt",
      autoencoder: "seedvr2_vae_f16.ckpt", modifier: .inpainting,
      objective: .u(conditionScale: 1000),
      noiseDiscretization: .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000)),
      latentsScalingFactor: 0.9152, hiresFixScale: 512,
      note:
        "[SeedVR2 3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) is ByteDance Seed's Apache 2.0-licensed one-step diffusion video restoration model. It uses adversarial post-training on real data and adaptive window attention for high-resolution restoration, targeting fast video enhancement from degraded inputs.",
      huggingFaceLink: "ByteDance-Seed/SeedVR2-3B"
    ),
    Specification(
      name: "SeedVR2 3B (8-bit S)", file: "seedvr2_3b_i8x.ckpt", prefix: "",
      version: .seedvr2_3b, defaultScale: 24, textEncoder: "seedvr2_3b_i8x.ckpt",
      autoencoder: "seedvr2_vae_f16.ckpt", modifier: .inpainting,
      objective: .u(conditionScale: 1000),
      noiseDiscretization: .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000)),
      latentsScalingFactor: 0.9152, hiresFixScale: 512,
      note:
        "[SeedVR2 3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) is ByteDance Seed's Apache 2.0-licensed one-step diffusion video restoration model. It uses adversarial post-training on real data and adaptive window attention for high-resolution restoration, targeting fast video enhancement from degraded inputs.",
      huggingFaceLink: "ByteDance-Seed/SeedVR2-3B"
    ),
    Specification(
      name: "SeedVR2 3B (6-bit)", file: "seedvr2_3b_q6p.ckpt", prefix: "",
      version: .seedvr2_3b, defaultScale: 24, textEncoder: "seedvr2_3b_q6p.ckpt",
      autoencoder: "seedvr2_vae_f16.ckpt", modifier: .inpainting,
      objective: .u(conditionScale: 1000),
      noiseDiscretization: .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000)),
      latentsScalingFactor: 0.9152, hiresFixScale: 512,
      note:
        "[SeedVR2 3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) is ByteDance Seed's Apache 2.0-licensed one-step diffusion video restoration model. It uses adversarial post-training on real data and adaptive window attention for high-resolution restoration, targeting fast video enhancement from degraded inputs."
    ),
    Specification(
      name: "SeedVR2 7B", file: "seedvr2_7b_q8p.ckpt", prefix: "",
      version: .seedvr2_7b, defaultScale: 24, textEncoder: "seedvr2_7b_q8p.ckpt",
      autoencoder: "seedvr2_vae_f16.ckpt", modifier: .inpainting,
      objective: .u(conditionScale: 1000),
      noiseDiscretization: .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000)),
      latentsScalingFactor: 0.9152, hiresFixScale: 512,
      note:
        "[SeedVR2 7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) is the larger ByteDance Seed Apache 2.0-licensed one-step diffusion video restoration model. It uses adversarial post-training on real data and adaptive window attention for high-resolution restoration, targeting fast video enhancement from degraded inputs.",
      huggingFaceLink: "ByteDance-Seed/SeedVR2-7B"
    ),
    Specification(
      name: "SeedVR2 7B (8-bit S)", file: "seedvr2_7b_i8x.ckpt", prefix: "",
      version: .seedvr2_7b, defaultScale: 24, textEncoder: "seedvr2_7b_i8x.ckpt",
      autoencoder: "seedvr2_vae_f16.ckpt", modifier: .inpainting,
      objective: .u(conditionScale: 1000),
      noiseDiscretization: .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000)),
      latentsScalingFactor: 0.9152, hiresFixScale: 512,
      note:
        "[SeedVR2 7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) is the larger ByteDance Seed Apache 2.0-licensed one-step diffusion video restoration model. It uses adversarial post-training on real data and adaptive window attention for high-resolution restoration, targeting fast video enhancement from degraded inputs.",
      huggingFaceLink: "ByteDance-Seed/SeedVR2-7B"
    ),
    Specification(
      name: "SeedVR2 7B (6-bit)", file: "seedvr2_7b_q6p.ckpt", prefix: "",
      version: .seedvr2_7b, defaultScale: 24, textEncoder: "seedvr2_7b_q6p.ckpt",
      autoencoder: "seedvr2_vae_f16.ckpt", modifier: .inpainting,
      objective: .u(conditionScale: 1000),
      noiseDiscretization: .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000)),
      latentsScalingFactor: 0.9152, hiresFixScale: 512,
      note:
        "[SeedVR2 7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) is the larger ByteDance Seed Apache 2.0-licensed one-step diffusion video restoration model. It uses adversarial post-training on real data and adaptive window attention for high-resolution restoration, targeting fast video enhancement from degraded inputs."
    ),
    Specification(
      name: "Z Image Turbo 1.0", file: "z_image_turbo_1.0_q8p.ckpt", prefix: "",
      version: .zImage, defaultScale: 16, textEncoder: "qwen_3_vl_4b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[Z Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) is a powerful and highly efficient image generation model with 6B parameters. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 8 sampling steps recommended.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Tongyi-MAI/Z-Image-Turbo"
    ),
    Specification(
      name: "Z Image Turbo 1.0 (8-bit S)", file: "z_image_turbo_1.0_i8x.ckpt", prefix: "",
      version: .zImage, defaultScale: 16, textEncoder: "qwen_3_vl_4b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[Z Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) is a powerful and highly efficient image generation model with 6B parameters. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 8 sampling steps recommended.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Tongyi-MAI/Z-Image-Turbo"
    ),
    Specification(
      name: "Z Image Turbo 1.0 (6-bit)", file: "z_image_turbo_1.0_q6p.ckpt", prefix: "",
      version: .zImage, defaultScale: 16, textEncoder: "qwen_3_vl_4b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[Z Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) is a powerful and highly efficient image generation model with 6B parameters. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 8 sampling steps recommended.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Z Image Base 1.0", file: "z_image_1.0_q8p.ckpt", prefix: "",
      version: .zImage, defaultScale: 16, textEncoder: "qwen_3_vl_4b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 1 : 32) }),
        activationProjScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 2 : 1) }),
        activationFfnProjUpScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 1 : 32) }),
        activationFfnScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 2 : 1) })),
      note:
        "[Z Image](https://huggingface.co/Tongyi-MAI/Z-Image) is a powerful and highly efficient image generation model with 6B parameters. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 28 to 50 sampling steps recommended.",
      copyright: "© 2026 Alibaba", huggingFaceLink: "Tongyi-MAI/Z-Image"
    ),
    Specification(
      name: "Z Image Base 1.0 (8-bit S)", file: "z_image_1.0_i8x.ckpt", prefix: "",
      version: .zImage, defaultScale: 16, textEncoder: "qwen_3_vl_4b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 1 : 32) }),
        activationProjScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 2 : 1) }),
        activationFfnProjUpScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 1 : 32) }),
        activationFfnScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 2 : 1) })),
      note:
        "[Z Image](https://huggingface.co/Tongyi-MAI/Z-Image) is a powerful and highly efficient image generation model with 6B parameters. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 28 to 50 sampling steps recommended.",
      copyright: "© 2026 Alibaba", huggingFaceLink: "Tongyi-MAI/Z-Image"
    ),
    Specification(
      name: "Z Image Base 1.0 (6-bit)", file: "z_image_1.0_q6p.ckpt", prefix: "",
      version: .zImage, defaultScale: 16, textEncoder: "qwen_3_vl_4b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 1 : 32) }),
        activationProjScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 2 : 1) }),
        activationFfnProjUpScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 1 : 32) }),
        activationFfnScaling: Dictionary(
          uniqueKeysWithValues: (0..<34).map { ($0, $0 < 2 ? 2 : 1) })),
      note:
        "[Z Image](https://huggingface.co/Tongyi-MAI/Z-Image) is a powerful and highly efficient image generation model with 6B parameters. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 28 to 50 sampling steps recommended.",
      copyright: "© 2026 Alibaba"
    ),
    Specification(
      name: "Qwen Image 2512", file: "qwen_image_2512_q8p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationProjScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnProjUpScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image 2512](https://huggingface.co/Qwen/Qwen-Image-2512) is the december update of Qwen Image model with improvements on enhanced human realism, finer natural detail and improved text rendering. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Qwen/Qwen-Image-2512"
    ),
    Specification(
      name: "Qwen Image 2512 (8-bit S)", file: "qwen_image_2512_i8x.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationProjScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnProjUpScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image 2512](https://huggingface.co/Qwen/Qwen-Image-2512) is the december update of Qwen Image model with improvements on enhanced human realism, finer natural detail and improved text rendering. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Qwen/Qwen-Image-2512"
    ),
    Specification(
      name: "Qwen Image 2512 (6-bit)", file: "qwen_image_2512_q6p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationProjScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnProjUpScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image 2512](https://huggingface.co/Qwen/Qwen-Image-2512) is the december update of Qwen Image model with improvements on enhanced human realism, finer natural detail and improved text rendering. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image 2512 (BF16)", file: "qwen_image_2512_bf16_q8p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      isBf16: true,
      note:
        "[Qwen Image 2512](https://huggingface.co/Qwen/Qwen-Image-2512) is the december update of Qwen Image model with improvements on enhanced human realism, finer natural detail and improved text rendering. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image 2512 (BF16, 8-bit S)", file: "qwen_image_2512_bf16_i8x.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      isBf16: true,
      note:
        "[Qwen Image 2512](https://huggingface.co/Qwen/Qwen-Image-2512) is the december update of Qwen Image model with improvements on enhanced human realism, finer natural detail and improved text rendering. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image 2512 (BF16, 6-bit)", file: "qwen_image_2512_bf16_q6p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      isBf16: true,
      note:
        "[Qwen Image 2512](https://huggingface.co/Qwen/Qwen-Image-2512) is the december update of Qwen Image model with improvements on enhanced human realism, finer natural detail and improved text rendering. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "FLUX.2 [klein] 4B", file: "flux_2_klein_4b_q8p.ckpt", prefix: "",
      version: .flux2_4b, defaultScale: 16, textEncoder: "qwen_3_4b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", modifier: .kontext, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 512, hiresFixScale: 24,
      note:
        "[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) is a 4 billion parameter rectified flow transformer capable of generating, editing and combining images based on text instructions.",
      copyright: "© 2026 Black Forest Labs", huggingFaceLink: "black-forest-labs/FLUX.2-klein-4B"
    ),
    Specification(
      name: "FLUX.2 [klein] 4B (8-bit S)", file: "flux_2_klein_4b_i8x.ckpt", prefix: "",
      version: .flux2_4b, defaultScale: 16, textEncoder: "qwen_3_4b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", modifier: .kontext, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 512, hiresFixScale: 24,
      note:
        "[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) is a 4 billion parameter rectified flow transformer capable of generating, editing and combining images based on text instructions.",
      copyright: "© 2026 Black Forest Labs", huggingFaceLink: "black-forest-labs/FLUX.2-klein-4B"
    ),
    Specification(
      name: "FLUX.2 [klein] 4B (6-bit)", file: "flux_2_klein_4b_q6p.ckpt", prefix: "",
      version: .flux2_4b, defaultScale: 16, textEncoder: "qwen_3_4b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", modifier: .kontext, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 512, hiresFixScale: 24,
      note:
        "[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) is a 4 billion parameter rectified flow transformer capable of generating, editing and combining images based on text instructions.",
      copyright: "© 2026 Black Forest Labs"
    ),
    Specification(
      name: "FLUX.2 [klein] 4B Base", file: "flux_2_klein_base_4b_q8p.ckpt", prefix: "",
      version: .flux2_4b, defaultScale: 16, textEncoder: "qwen_3_4b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", modifier: .kontext, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 512, hiresFixScale: 24,
      note:
        "[FLUX.2 [klein] 4B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) is a 4 billion parameter rectified flow transformer capable of generating, editing and combining images based on text instructions.",
      copyright: "© 2026 Black Forest Labs",
      huggingFaceLink: "black-forest-labs/FLUX.2-klein-base-4B"
    ),
    Specification(
      name: "FLUX.2 [klein] 4B Base (8-bit S)", file: "flux_2_klein_base_4b_i8x.ckpt", prefix: "",
      version: .flux2_4b, defaultScale: 16, textEncoder: "qwen_3_4b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", modifier: .kontext, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 512, hiresFixScale: 24,
      note:
        "[FLUX.2 [klein] 4B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) is a 4 billion parameter rectified flow transformer capable of generating, editing and combining images based on text instructions.",
      copyright: "© 2026 Black Forest Labs",
      huggingFaceLink: "black-forest-labs/FLUX.2-klein-base-4B"
    ),
    Specification(
      name: "FLUX.2 [klein] 4B Base (6-bit)", file: "flux_2_klein_base_4b_q6p.ckpt", prefix: "",
      version: .flux2_4b, defaultScale: 16, textEncoder: "qwen_3_4b_q8p.ckpt",
      autoencoder: "flux_2_vae_f16.ckpt", modifier: .kontext, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 512, hiresFixScale: 24,
      note:
        "[FLUX.2 [klein] 4B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) is a 4 billion parameter rectified flow transformer capable of generating, editing and combining images based on text instructions.",
      copyright: "© 2026 Black Forest Labs"
    ),
    Specification(
      name: "Qwen Image 1.0", file: "qwen_image_1.0_q8p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[Qwen Image](https://huggingface.co/Qwen/Qwen-Image) is a state-of-the-art open-source image generation model known for its exceptional text layout and prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Qwen/Qwen-Image"
    ),
    Specification(
      name: "Qwen Image 1.0 (6-bit)", file: "qwen_image_1.0_q6p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[Qwen Image](https://huggingface.co/Qwen/Qwen-Image) is a state-of-the-art open-source image generation model known for its exceptional text layout and prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image 1.0 (BF16)", file: "qwen_image_1.0_bf16_q8p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24, isBf16: true,
      note:
        "[Qwen Image](https://huggingface.co/Qwen/Qwen-Image) is a state-of-the-art open-source image generation model known for its exceptional text layout and prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image 1.0 (BF16, 6-bit)", file: "qwen_image_1.0_bf16_q6p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24, isBf16: true,
      note:
        "[Qwen Image](https://huggingface.co/Qwen/Qwen-Image) is a state-of-the-art open-source image generation model known for its exceptional text layout and prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image Edit 2511", file: "qwen_image_edit_2511_q8p.ckpt",
      prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEdit2511,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationProjScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) is an enhanced image editing model that significantly improves character consistency, mitigates image drift, and strengthens multi-person fusion capabilities compared to its predecessor (2509). It integrates popular LoRA features natively, enabling advanced lighting control and viewpoint generation without extra tuning, alongside specialized industrial design and geometric reasoning capabilities. It is Apache 2.0-licensed, with 40 inference steps recommended for optimal results.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Qwen/Qwen-Image-Edit-2511"
    ),
    Specification(
      name: "Qwen Image Edit 2511 (8-bit S)", file: "qwen_image_edit_2511_i8x.ckpt",
      prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEdit2511,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationProjScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) is an enhanced image editing model that significantly improves character consistency, mitigates image drift, and strengthens multi-person fusion capabilities compared to its predecessor (2509). It integrates popular LoRA features natively, enabling advanced lighting control and viewpoint generation without extra tuning, alongside specialized industrial design and geometric reasoning capabilities. It is Apache 2.0-licensed, with 40 inference steps recommended for optimal results.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Qwen/Qwen-Image-Edit-2511"
    ),
    Specification(
      name: "Qwen Image Edit 2511 (6-bit)", file: "qwen_image_edit_2511_q6p.ckpt",
      prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEdit2511,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationProjScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) is an enhanced image editing model that significantly improves character consistency, mitigates image drift, and strengthens multi-person fusion capabilities compared to its predecessor (2509). It integrates popular LoRA features natively, enabling advanced lighting control and viewpoint generation without extra tuning, alongside specialized industrial design and geometric reasoning capabilities. It is Apache 2.0-licensed, with 40 inference steps recommended for optimal results.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image Edit 2511 (BF16)", file: "qwen_image_edit_2511_bf16_q8p.ckpt",
      prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEdit2511,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24, isBf16: true,
      note:
        "[Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) is an enhanced image editing model that significantly improves character consistency, mitigates image drift, and strengthens multi-person fusion capabilities compared to its predecessor (2509). It integrates popular LoRA features natively, enabling advanced lighting control and viewpoint generation without extra tuning, alongside specialized industrial design and geometric reasoning capabilities. It is Apache 2.0-licensed, with 40 inference steps recommended for optimal results. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image Edit 2511 (BF16, 8-bit S)", file: "qwen_image_edit_2511_bf16_i8x.ckpt",
      prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEdit2511,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24, isBf16: true,
      note:
        "[Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) is an enhanced image editing model that significantly improves character consistency, mitigates image drift, and strengthens multi-person fusion capabilities compared to its predecessor (2509). It integrates popular LoRA features natively, enabling advanced lighting control and viewpoint generation without extra tuning, alongside specialized industrial design and geometric reasoning capabilities. It is Apache 2.0-licensed, with 40 inference steps recommended for optimal results. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image Edit 2511 (BF16, 6-bit)", file: "qwen_image_edit_2511_bf16_q6p.ckpt",
      prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEdit2511,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24, isBf16: true,
      note:
        "[Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) is an enhanced image editing model that significantly improves character consistency, mitigates image drift, and strengthens multi-person fusion capabilities compared to its predecessor (2509). It integrates popular LoRA features natively, enabling advanced lighting control and viewpoint generation without extra tuning, alongside specialized industrial design and geometric reasoning capabilities. It is Apache 2.0-licensed, with 40 inference steps recommended for optimal results. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image Layered 1.0 (BF16)", file: "qwen_image_layered_1.0_bf16_q8p.ckpt",
      prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_layered_vae_f16.ckpt", modifier: .qwenimageLayered,
      objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 4) })),
      isBf16: true,
      note:
        "[Qwen Image Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) is a specialized model capable of decomposing an image into multiple transparent RGBA layers to unlock inherent editability. By physically isolating semantic components, it enables high-fidelity operations such as resizing, repositioning, and recoloring without affecting the rest of the image. It is Apache 2.0-licensed and commercially friendly. The model supports flexible and recursive decomposition, allowing users to define specific layer counts (Batch Size), with a recommended resolution of 640px and 50 inference steps. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Qwen/Qwen-Image-Layered"
    ),
    Specification(
      name: "Qwen Image Layered 1.0 (BF16, 6-bit)", file: "qwen_image_layered_1.0_bf16_q6p.ckpt",
      prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_layered_vae_f16.ckpt", modifier: .qwenimageLayered,
      objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationQkScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 4) })),
      isBf16: true,
      note:
        "[Qwen Image Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) is a specialized model capable of decomposing an image into multiple transparent RGBA layers to unlock inherent editability. By physically isolating semantic components, it enables high-fidelity operations such as resizing, repositioning, and recoloring without affecting the rest of the image. It is Apache 2.0-licensed and commercially friendly. The model supports flexible and recursive decomposition, allowing users to define specific layer counts (Batch Size), with a recommended resolution of 640px and 50 inference steps. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image Edit 2509", file: "qwen_image_edit_2509_q8p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEditPlus,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationProjScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image Edit 2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) is a state-of-the-art open-source image edit model excels at image edit tasks such as background alternation, style transfer, object removal etc. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended. This is an update in Sep, 2025.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Qwen/Qwen-Image-Edit-2509"
    ),
    Specification(
      name: "Qwen Image Edit 2509 (6-bit)", file: "qwen_image_edit_2509_q6p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEditPlus,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationProjScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image Edit 2509](https://huggingface.co/Qwen/Qwen-Image-2509) is a state-of-the-art open-source image edit model excels at image edit tasks such as background alternation, style transfer, object removal etc. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended. This is an update in Sep, 2025.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image Edit 2509 (BF16)", file: "qwen_image_edit_2509_bf16_q8p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEditPlus,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationProjScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) }),
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      isBf16: true,
      note:
        "[Qwen Image Edit 2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) is a state-of-the-art open-source image edit model excels at image edit tasks such as background alternation, style transfer, object removal etc. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended. This is an update in Sep, 2025. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image Edit 2509 (BF16, 6-bit)", file: "qwen_image_edit_2509_bf16_q6p.ckpt",
      prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .qwenimageEditPlus,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      isBf16: true,
      note:
        "[Qwen Image Edit 2509](https://huggingface.co/Qwen/Qwen-Image-2509) is a state-of-the-art open-source image edit model excels at image edit tasks such as background alternation, style transfer, object removal etc. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended. This is an update in Sep, 2025. The BF16 version is only compatible with macOS 15, iOS 18 and above.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Qwen Image Edit 1.0", file: "qwen_image_edit_1.0_q8p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) is a state-of-the-art open-source image edit model excels at image edit tasks such as background alternation, style transfer, object removal etc. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Qwen/Qwen-Image-Edit"
    ),
    Specification(
      name: "Qwen Image Edit 1.0 (6-bit)", file: "qwen_image_edit_1.0_q6p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(
        qkNorm: true, dualAttentionLayers: [],
        activationFfnScaling: Dictionary(uniqueKeysWithValues: (0..<60).map { ($0, 2) })),
      note:
        "[Qwen Image Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) is a state-of-the-art open-source image edit model excels at image edit tasks such as background alternation, style transfer, object removal etc. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "HiDream I1 [fast]", file: "hidream_i1_fast_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      isConsistencyModel: true, objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [fast]](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 10–20 sampling steps recommended. Text guidance is not effective for this model.",
      huggingFaceLink: "HiDream-ai/HiDream-I1-Fast"
    ),
    Specification(
      name: "HiDream I1 [fast] (8-bit S)", file: "hidream_i1_fast_i8x.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      isConsistencyModel: true, objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [fast]](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 10–20 sampling steps recommended. Text guidance is not effective for this model.",
      huggingFaceLink: "HiDream-ai/HiDream-I1-Fast"
    ),
    Specification(
      name: "HiDream I1 [fast] (5-bit)", file: "hidream_i1_fast_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      isConsistencyModel: true, objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [fast]](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 10–20 sampling steps recommended. Text guidance is not effective for this model."
    ),
    Specification(
      name: "HiDream I1 [dev]", file: "hidream_i1_dev_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), guidanceEmbed: true, paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [dev]](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 20–30 sampling steps recommended. Text guidance is not effective for this model.",
      huggingFaceLink: "HiDream-ai/HiDream-I1-Dev"
    ),
    Specification(
      name: "HiDream I1 [dev] (8-bit S)", file: "hidream_i1_dev_i8x.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), guidanceEmbed: true, paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [dev]](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 20–30 sampling steps recommended. Text guidance is not effective for this model.",
      huggingFaceLink: "HiDream-ai/HiDream-I1-Dev"
    ),
    Specification(
      name: "HiDream I1 [dev] (5-bit)", file: "hidream_i1_dev_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), guidanceEmbed: true, paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [dev]](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 20–30 sampling steps recommended. Text guidance is not effective for this model."
    ),
    Specification(
      name: "HiDream I1 [full]", file: "hidream_i1_full_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-I1 [full]](https://huggingface.co/HiDream-ai/HiDream-I1-Full) is a state-of-the-art open-source image generation model known for its exceptional prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended.",
      huggingFaceLink: "HiDream-ai/HiDream-I1-Full"
    ),
    Specification(
      name: "HiDream I1 [full] (8-bit S)", file: "hidream_i1_full_i8x.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-I1 [full]](https://huggingface.co/HiDream-ai/HiDream-I1-Full) is a state-of-the-art open-source image generation model known for its exceptional prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended.",
      huggingFaceLink: "HiDream-ai/HiDream-I1-Full"
    ),
    Specification(
      name: "HiDream I1 [full] (5-bit)", file: "hidream_i1_full_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-I1 [full]](https://huggingface.co/HiDream-ai/HiDream-I1-Full) is a state-of-the-art open-source image generation model known for its exceptional prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended."
    ),
    Specification(
      name: "HiDream E1-1", file: "hidream_e1_1_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", modifier: .editing,
      clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) is an image editing model built on HiDream-I1. It is MIT-licensed and commercially friendly. Trained with dynamic resolutions (around 1MP) using a Flow Matching objective, the model performs best with trailing samplers and 30–50 sampling steps.",
      huggingFaceLink: "HiDream-ai/HiDream-E1-1"
    ),
    Specification(
      name: "HiDream E1-1 (5-bit)", file: "hidream_e1_1_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", modifier: .editing,
      clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) is an image editing model built on HiDream-I1. It is MIT-licensed and commercially friendly. Trained with dynamic resolutions (around 1MP) using a Flow Matching objective, the model performs best with trailing samplers and 30–50 sampling steps."
    ),
    Specification(
      name: "HiDream E1 [full]", file: "hidream_e1_full_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 12, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", modifier: .editing,
      clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-E1 [full]](https://huggingface.co/HiDream-ai/HiDream-E1-Full) is an image editing model built on HiDream-I1. It is MIT-licensed and commercially friendly. Trained at 768×768 resolution using a Flow Matching objective, the model performs best with trailing samplers and 30–50 sampling steps. For optimal results, ensure the width is set to 768 and use the following prompt format: Editing Instruction: {}. Target Image Description: {}.",
      huggingFaceLink: "HiDream-ai/HiDream-E1-Full"
    ),
    Specification(
      name: "HiDream E1 [full] (5-bit)", file: "hidream_e1_full_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 12, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", modifier: .editing,
      clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-E1 [full]](https://huggingface.co/HiDream-ai/HiDream-E1-Full) is an image editing model built on HiDream-I1. It is MIT-licensed and commercially friendly. Trained at 768×768 resolution using a Flow Matching objective, the model performs best with trailing samplers and 30–50 sampling steps. For optimal results, ensure the width is set to 768 and use the following prompt format: Editing Instruction: {}. Target Image Description: {}."
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert T2V A14B", file: "wan_v2.2_a14b_hne_t2v_q8p.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Wan-AI/Wan2.2-T2V-A14B"
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert T2V A14B (8-bit S)", file: "wan_v2.2_a14b_hne_t2v_i8x.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Wan-AI/Wan2.2-T2V-A14B"
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert T2V A14B (6-bit, SVDQuant)",
      file: "wan_v2.2_a14b_hne_t2v_q6p_svd.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16, builtinLora: true,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert T2V A14B", file: "wan_v2.2_a14b_lne_t2v_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert T2V A14B (8-bit S)", file: "wan_v2.2_a14b_lne_t2v_i8x.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert T2V A14B (6-bit, SVDQuant)",
      file: "wan_v2.2_a14b_lne_t2v_q6p_svd.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16, builtinLora: true,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert I2V A14B", file: "wan_v2.2_a14b_hne_i2v_q8p.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Wan-AI/Wan2.2-I2V-A14B"
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert I2V A14B (8-bit S)", file: "wan_v2.2_a14b_hne_i2v_i8x.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Wan-AI/Wan2.2-I2V-A14B"
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert I2V A14B (6-bit, SVDQuant)",
      file: "wan_v2.2_a14b_hne_i2v_q6p_svd.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      builtinLora: true,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert I2V A14B", file: "wan_v2.2_a14b_lne_i2v_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert I2V A14B (8-bit S)", file: "wan_v2.2_a14b_lne_i2v_i8x.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert I2V A14B (6-bit, SVDQuant)",
      file: "wan_v2.2_a14b_lne_i2v_q6p_svd.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      builtinLora: true,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.2 TI2V 5B", file: "wan_v2.2_5b_ti2v_f16.ckpt",
      prefix: "",
      version: .wan22_5b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.2_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: nil, framesPerSecond: 24,
      note:
        "[Wan2.2 TI2V 5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) is a state-of-the-art text-image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 121 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Wan-AI/Wan2.2-TI2V-5B"
    ),
    Specification(
      name: "Wan 2.2 TI2V 5B (8-bit)", file: "wan_v2.2_5b_ti2v_q8p.ckpt",
      prefix: "",
      version: .wan22_5b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.2_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: nil, framesPerSecond: 24,
      note:
        "[Wan2.2 TI2V 5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) is a state-of-the-art text-image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 121 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.1 T2V 1.3B", file: "wan_v2.1_1.3b_480p_f16.ckpt", prefix: "",
      version: .wan21_1_3b, defaultScale: 8, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 12,
      teaCacheCoefficients: [
        -5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 6.0. For best results, set Text Guidance above 5.0. Wan2.1 is trained with a Flow Matching objective, and trailing samplers will produce the best outputs.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Wan-AI/Wan2.1-T2V-1.3B"
    ),
    Specification(
      name: "Wan 2.1 T2V 1.3B (8-bit)", file: "wan_v2.1_1.3b_480p_q8p.ckpt", prefix: "",
      version: .wan21_1_3b, defaultScale: 8, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 12,
      teaCacheCoefficients: [
        -5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 6.0. For best results, set Text Guidance above 5.0. Wan2.1 is trained with a Flow Matching objective, and trailing samplers will produce the best outputs.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.1 T2V 14B", file: "wan_v2.1_14b_720p_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 5.0. For best results, set Text Guidance above 5.0. Wan2.1 is trained with a Flow Matching objective, and trailing samplers will produce the best outputs.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Wan-AI/Wan2.1-T2V-14B"
    ),
    Specification(
      name: "Wan 2.1 T2V 14B (6-bit, SVDQuant)", file: "wan_v2.1_14b_720p_q6p_svd.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16, builtinLora: true,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.1 T2V 14B (5-bit, SVDQuant)", file: "wan_v2.1_14b_720p_q5p_svd.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", deprecated: true, hiresFixScale: 16,
      builtinLora: true,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.1 I2V 14B 480p", file: "wan_v2.1_14b_i2v_480p_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 8, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting,
      clipEncoder: "open_clip_xlm_roberta_large_vit_h14_f16.ckpt", hiresFixScale: 12,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 I2V 14B 480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Wan-AI/Wan2.1-I2V-14B-480P"
    ),
    Specification(
      name: "Wan 2.1 I2V 14B 480p (6-bit, SVDQuant)", file: "wan_v2.1_14b_i2v_480p_q6p_svd.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 8, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting,
      clipEncoder: "open_clip_xlm_roberta_large_vit_h14_f16.ckpt", hiresFixScale: 12,
      builtinLora: true,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 I2V 14B 480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "Wan 2.1 I2V 14B 720p", file: "wan_v2.1_14b_i2v_720p_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting,
      clipEncoder: "open_clip_xlm_roberta_large_vit_h14_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 I2V 14B 720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba", huggingFaceLink: "Wan-AI/Wan2.1-I2V-14B-720P"
    ),
    Specification(
      name: "Wan 2.1 I2V 14B 720p (6-bit, SVDQuant)", file: "wan_v2.1_14b_i2v_720p_q6p_svd.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting,
      clipEncoder: "open_clip_xlm_roberta_large_vit_h14_f16.ckpt", hiresFixScale: 16,
      builtinLora: true,
      teaCacheCoefficients: [
        8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 I2V 14B 720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0.",
      copyright: "© 2025 Alibaba"
    ),
    Specification(
      name: "FLUX.1 [schnell]", file: "flux_1_schnell_q8p.ckpt", prefix: "",
      version: .flux1, defaultScale: 16, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt",
      highPrecisionAutoencoder: true, isConsistencyModel: true, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 256, hiresFixScale: 24,
      huggingFaceLink: "black-forest-labs/FLUX.1-schnell"),
    Specification(
      name: "FLUX.1 [schnell] (5-bit)", file: "flux_1_schnell_q5p.ckpt", prefix: "",
      version: .flux1, defaultScale: 16, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt",
      highPrecisionAutoencoder: true, isConsistencyModel: true, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 256, hiresFixScale: 24),
    Specification(
      name: "PixArt Sigma XL 1K", file: "pixart_sigma_xl_2_1024_ms_f16.ckpt", prefix: "",
      version: .pixart, defaultScale: 16, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt"),
    Specification(
      name: "PixArt Sigma XL 1K (8-bit)", file: "pixart_sigma_xl_2_1024_ms_q8p.ckpt", prefix: "",
      version: .pixart, defaultScale: 16, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt"),
    Specification(
      name: "PixArt Sigma XL 512", file: "pixart_sigma_xl_2_512_ms_f16.ckpt", prefix: "",
      version: .pixart, defaultScale: 8, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt"),
    Specification(
      name: "PixArt Sigma XL 512 (8-bit)", file: "pixart_sigma_xl_2_512_ms_q8p.ckpt", prefix: "",
      version: .pixart, defaultScale: 8, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt"),
    Specification(
      name: "AuraFlow v0.1", file: "auraflow_v0.1_q8p.ckpt", prefix: "",
      version: .auraflow, defaultScale: 16, textEncoder: "pile_t5_xl_encoder_q8p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", deprecated: true, objective: .u(conditionScale: 1000)),
    Specification(
      name: "AuraFlow v0.1 (8-bit)", file: "auraflow_v0.1_q5p.ckpt", prefix: "",
      version: .auraflow, defaultScale: 16, textEncoder: "pile_t5_xl_encoder_q8p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", deprecated: true, objective: .u(conditionScale: 1000)),
    Specification(
      name: "AuraFlow v0.2", file: "auraflow_v0.2_q8p.ckpt", prefix: "",
      version: .auraflow, defaultScale: 16, textEncoder: "pile_t5_xl_encoder_q8p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", objective: .u(conditionScale: 1000)),
    Specification(
      name: "AuraFlow v0.2 (8-bit)", file: "auraflow_v0.2_q5p.ckpt", prefix: "",
      version: .auraflow, defaultScale: 16, textEncoder: "pile_t5_xl_encoder_q8p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", objective: .u(conditionScale: 1000)),
    Specification(
      name: "SDXL Base (v1.0)", file: "sd_xl_base_1.0_f16.ckpt", prefix: "", version: .sdxlBase,
      defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Base v1.0 (8-bit)", file: "sd_xl_base_1.0_q6p_q8p.ckpt", prefix: "",
      version: .sdxlBase,
      defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Refiner (v1.0)", file: "sd_xl_refiner_1.0_f16.ckpt", prefix: "",
      version: .sdxlRefiner, defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Refiner v1.0 (8-bit)", file: "sd_xl_refiner_1.0_q6p_q8p.ckpt", prefix: "",
      version: .sdxlRefiner, defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Base (v0.9)", file: "sd_xl_base_0.9_f16.ckpt", prefix: "", version: .sdxlBase,
      defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_f16.ckpt", deprecated: true, clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Refiner (v0.9)", file: "sd_xl_refiner_0.9_f16.ckpt", prefix: "",
      version: .sdxlRefiner, defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_f16.ckpt", deprecated: true, clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "Fooocus Inpaint SDXL v2.6", file: "fooocus_inpaint_sd_xl_v2.6_f16.ckpt", prefix: "",
      version: .sdxlBase, defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", modifier: .inpainting,
      clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "Fooocus Inpaint SDXL v2.6 (8-bit)", file: "fooocus_inpaint_sd_xl_v2.6_q6p_q8p.ckpt",
      prefix: "", version: .sdxlBase, defaultScale: 16,
      textEncoder: "open_clip_vit_bigg14_f16.ckpt", autoencoder: "sdxl_vae_v1.0_f16.ckpt",
      modifier: .inpainting, clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "Stable Diffusion v1.4", file: "sd_v1.4_f16.ckpt", prefix: "", version: .v1,
      deprecated: true),
    Specification(
      name: "Stable Diffusion v1.5", file: "sd_v1.5_f16.ckpt", prefix: "", version: .v1),
    Specification(
      name: "Stable Diffusion v1.5 Inpainting", file: "sd_v1.5_inpainting_f16.ckpt",
      prefix: "", version: .v1, modifier: .inpainting),
    Specification(
      name: "Stable Diffusion v2.0", file: "sd_v2.0_f16.ckpt", prefix: "", version: .v2,
      textEncoder: "open_clip_vit_h14_f16.ckpt", deprecated: true),
    Specification(
      name: "Stable Diffusion v2.0 768-v", file: "sd_v2.0_768_v_f16.ckpt", prefix: "",
      version: .v2, defaultScale: 12, textEncoder: "open_clip_vit_h14_f16.ckpt", deprecated: true,
      objective: .v),
    Specification(
      name: "Stable Diffusion v2.0 Inpainting", file: "sd_v2.0_inpainting_f16.ckpt",
      prefix: "", version: .v2, textEncoder: "open_clip_vit_h14_f16.ckpt", modifier: .inpainting),
    Specification(
      name: "Stable Diffusion v2.0 Depth", file: "sd_v2.0_depth_f16.ckpt",
      prefix: "", version: .v2, textEncoder: "open_clip_vit_h14_f16.ckpt", modifier: .depth),
    Specification(
      name: "Stable Diffusion v2.1", file: "sd_v2.1_f16.ckpt", prefix: "", version: .v2,
      textEncoder: "open_clip_vit_h14_f16.ckpt"),
    Specification(
      name: "Stable Diffusion v2.1 768-v", file: "sd_v2.1_768_v_f16.ckpt", prefix: "",
      version: .v2, upcastAttention: true, defaultScale: 12,
      textEncoder: "open_clip_vit_h14_f16.ckpt", objective: .v),
    Specification(
      name: "Kandinsky v2.1", file: "kandinsky_f16.ckpt", prefix: "",
      version: .kandinsky21, upcastAttention: false, defaultScale: 12,
      textEncoder: "xlm_roberta_f16.ckpt", autoencoder: "kandinsky_movq_f16.ckpt",
      deprecated: true, imageEncoder: "image_vit_l14_f16.ckpt",
      clipEncoder: "clip_vit_l14_f16.ckpt",
      diffusionMapping: "kandinsky_diffusion_mapping_f16.ckpt"),
    Specification(
      name: "Stable Video Diffusion I2V XT v1.0", file: "svd_i2v_xt_1.0_f16.ckpt", prefix: "",
      version: .svdI2v,
      defaultScale: 8, textEncoder: "open_clip_vit_h14_vision_model_f16.ckpt", deprecated: true,
      clipEncoder: "svd_i2v_xt_1.0_f16.ckpt", conditioning: .noise, objective: .v,
      noiseDiscretization: .edm(.init(sigmaMax: 700.0))),
    Specification(
      name: "Stable Video Diffusion I2V XT 1.0 (8-bit)", file: "svd_i2v_xt_1.0_q6p_q8p.ckpt",
      prefix: "",
      version: .svdI2v,
      defaultScale: 8, textEncoder: "open_clip_vit_h14_vision_model_f16.ckpt", deprecated: true,
      clipEncoder: "svd_i2v_xt_1.0_q6p_q8p.ckpt", conditioning: .noise, objective: .v,
      noiseDiscretization: .edm(.init(sigmaMax: 700.0))),
    Specification(
      name: "Stable Video Diffusion I2V XT v1.1 (8-bit)", file: "svd_i2v_xt_1.1_q6p_q8p.ckpt",
      prefix: "",
      version: .svdI2v,
      defaultScale: 8, textEncoder: "open_clip_vit_h14_vision_model_f16.ckpt",
      clipEncoder: "svd_i2v_xt_1.1_q6p_q8p.ckpt", conditioning: .noise, objective: .v,
      noiseDiscretization: .edm(.init(sigmaMax: 700.0))),
    Specification(
      name: "AnimateLCM SVD XT v1.1 (8-bit)", file: "animatelcm_svd_xt_v1.1_q6p_q8p.ckpt",
      prefix: "",
      version: .svdI2v,
      defaultScale: 8, textEncoder: "open_clip_vit_h14_vision_model_f16.ckpt",
      clipEncoder: "open_clip_vit_h14_visual_proj_f16.ckpt", conditioning: .noise, objective: .v,
      noiseDiscretization: .edm(.init(sigmaMax: 700.0))),
    Specification(
      name: "Stable Video Diffusion I2V v1.0 (8-bit)", file: "svd_i2v_1.0_q6p_q8p.ckpt",
      prefix: "",
      version: .svdI2v,
      defaultScale: 8, textEncoder: "open_clip_vit_h14_vision_model_f16.ckpt",
      clipEncoder: "svd_i2v_1.0_q6p_q8p.ckpt", conditioning: .noise, objective: .v,
      noiseDiscretization: .edm(.init(sigmaMax: 700.0))),
    Specification(
      name: "MiniSD v1.4", file: "minisd_v1.4_f16.ckpt", prefix: "", version: .v1,
      defaultScale: 4, deprecated: true),
    Specification(
      name: "Instruct Pix2Pix", file: "instruct_pix2pix_22000_f16.ckpt", prefix: "",
      version: .v1, defaultScale: 8, modifier: .editing, deprecated: true),
  ]

  private static let builtinModelsAndAvailableSpecifications: (Set<String>, [Specification]) = {
    let jsonFile = filePathForModelDownloaded("custom.json")
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

  public static func availableSpecificationForTriggerWord(_ triggerWord: String) -> Specification? {
    let cleanupTriggerWord = String(triggerWord.lowercased().filter { $0.isLetter || $0.isNumber })
    for specification in availableSpecifications {
      if String(specification.name.lowercased().filter { $0.isLetter || $0.isNumber }).contains(
        cleanupTriggerWord)
      {
        return specification
      }
    }
    return nil
  }

  public static func sortCustomSpecifications() {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = ModelZoo.filePathForOtherModelDownloaded("custom.json")
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

  public static func appendCustomSpecification(_ specification: Specification) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = ModelZoo.filePathForOtherModelDownloaded("custom.json")
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
    // Still respect the order.
    availableSpecifications.append(specification)
    self.availableSpecifications = availableSpecifications
    specificationMapping[specification.file] = specification
  }

  public static func updateCustomSpecification(_ specification: Specification) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = ModelZoo.filePathForOtherModelDownloaded("custom.json")

    // Load existing specifications from custom.json
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      if let jsonSpecification = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value }) {
        customSpecifications.append(contentsOf: jsonSpecification)
      }
    }

    // Find the index of the specification with matching file name
    if let index = customSpecifications.firstIndex(where: { $0.name == specification.name }) {
      // Replace the existing specification with the updated one
      customSpecifications[index] = specification
    } else {
      // If no matching specification was found, just append it
      customSpecifications.append(specification)
    }

    // Write updated specifications back to custom.json
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    guard let jsonData = try? jsonEncoder.encode(customSpecifications) else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)

    // Update the in-memory cache
    var availableSpecifications = self.availableSpecifications
    // Replace or add the specification in the array
    if let index = availableSpecifications.firstIndex(where: { $0.file == specification.file }) {
      availableSpecifications[index] = specification
    } else {
      availableSpecifications.append(specification)
    }
    self.availableSpecifications = availableSpecifications

    // Update the mapping dictionary
    overrideMapping[specification.file] = specification
    specificationMapping[specification.file] = specification
  }

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in availableSpecifications {
      mapping[specification.file] = specification
    }
    return mapping
  }()

  public static var anyModelDownloaded: String? {
    let availableSpecifications = availableSpecifications
    for specification in availableSpecifications {
      if isModelDownloaded(specification) {
        return specification.file
      }
    }
    return nil
  }

  private static func filePathForOtherModelDownloaded(_ name: String) -> String {
    let modelZooUrl = persistentModelsDirectory()
    if isExternalUrlsPreferred, let externalUrl = externalUrls.first {
      return externalUrl.appendingPathComponent(name).path
    }
    return modelZooUrl.appendingPathComponent(name).path
  }

  public static var isExternalUrlsPreferred: Bool = false

  public static var externalUrls: [URL] = [URL]() {
    didSet {
      #if (os(macOS) || (os(iOS) && targetEnvironment(macCatalyst)))
        guard oldValue != externalUrls else { return }
        for url in oldValue {
          url.stopAccessingSecurityScopedResource()
        }
        for url in externalUrls {
          let _ = url.startAccessingSecurityScopedResource()
        }
      #endif
    }
  }

  public static func specificationForHumanReadableModel(_ name: String) -> Specification? {
    return availableSpecifications.first { $0.name == name }
  }

  public enum ModelReferenceMatchKind: String, Codable {
    case file
    case humanReadableName
    case huggingFaceRepo
    case huggingFaceURL
  }

  public struct ModelReferenceResolution {
    public var specification: Specification
    public var normalizedInput: String
    public var kind: ModelReferenceMatchKind
    public var normalizedHuggingFaceRepo: String?
    public init(
      specification: Specification, normalizedInput: String, kind: ModelReferenceMatchKind,
      normalizedHuggingFaceRepo: String? = nil
    ) {
      self.specification = specification
      self.normalizedInput = normalizedInput
      self.kind = kind
      self.normalizedHuggingFaceRepo = normalizedHuggingFaceRepo
    }
  }

  // Repo aliases for rename / organization migration cases.
  // Key and value are both expected as "owner/repo" format.
  public static var huggingFaceRepoAliases: [String: String] = [:]

  // Highest-priority explicit mapping for HF repo to model spec.
  public static var huggingFaceRepoOverrideMapping: [String: Specification] = [:]

  // Last-chance explicit mapping for HF repo to model spec.
  public static var huggingFaceRepoFallbackMapping: [String: Specification] = [:]

  public static func normalizeHuggingFaceRepo(_ input: String) -> String? {
    let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    if trimmed.hasPrefix("hf://") {
      return sanitizeHuggingFaceRepo(String(trimmed.dropFirst("hf://".count)))
    }
    if let url = URL(string: trimmed),
      let host = url.host?.lowercased(),
      host.contains("huggingface.co")
    {
      let parts = url.path.split(separator: "/").map(String.init)
      let startIndex: Int
      if let first = parts.first, ["models", "datasets", "spaces"].contains(first.lowercased()) {
        startIndex = 1
      } else {
        startIndex = 0
      }
      guard parts.count >= startIndex + 2 else { return nil }
      return sanitizeHuggingFaceRepo("\(parts[startIndex])/\(parts[startIndex + 1])")
    }
    if trimmed.contains("/"), !trimmed.contains("://"), !trimmed.contains(" ") {
      return sanitizeHuggingFaceRepo(trimmed)
    }
    return nil
  }

  public static func specificationForHuggingFaceRepo(_ repo: String) -> Specification? {
    guard let canonicalRepo = canonicalizedRepoForLookup(repo) else { return nil }
    if let override = valueForCanonicalizedRepoKey(
      huggingFaceRepoOverrideMapping, key: canonicalRepo)
    {
      return override
    }
    if let matched = availableSpecifications.first(where: {
      guard let link = $0.huggingFaceLink, let canonicalLink = canonicalHuggingFaceRepo(link) else {
        return false
      }
      return canonicalLink == canonicalRepo
    }) {
      return matched
    }
    return valueForCanonicalizedRepoKey(huggingFaceRepoFallbackMapping, key: canonicalRepo)
  }

  public static func resolveModelReference(_ input: String) -> ModelReferenceResolution? {
    let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    if let spec = specificationForModel(trimmed) {
      return ModelReferenceResolution(
        specification: spec, normalizedInput: trimmed, kind: .file)
    }
    if let spec = specificationForHumanReadableModel(trimmed) {
      return ModelReferenceResolution(
        specification: spec, normalizedInput: trimmed, kind: .humanReadableName)
    }
    if let repo = normalizeHuggingFaceRepo(trimmed),
      let spec = specificationForHuggingFaceRepo(repo)
    {
      return ModelReferenceResolution(
        specification: spec, normalizedInput: trimmed,
        kind: huggingFaceURLLike(trimmed) ? .huggingFaceURL : .huggingFaceRepo,
        normalizedHuggingFaceRepo: canonicalizedRepoForLookup(repo))
    }
    return nil
  }

  public static func candidateSpecifications(
    forModelReference input: String, limit: Int = 5
  ) -> [Specification] {
    let query = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !query.isEmpty else { return [] }
    let normalizedRepo = normalizeHuggingFaceRepo(input)?.lowercased()
    var scored = [(score: Int, specification: Specification)]()
    for specification in availableSpecifications where specification.remoteApiModelConfig == nil {
      let file = specification.file.lowercased()
      let name = specification.name.lowercased()
      let huggingFace = specification.huggingFaceLink?.lowercased() ?? ""
      var score = 0
      if file == query || name == query || huggingFace == query {
        score = 1000
      } else if file.hasPrefix(query) || name.hasPrefix(query) || huggingFace.hasPrefix(query) {
        score = 700
      } else if file.contains(query) || name.contains(query) || huggingFace.contains(query) {
        score = 500
      }
      if let normalizedRepo {
        if huggingFace == normalizedRepo {
          score = max(score, 950)
        } else if !huggingFace.isEmpty,
          huggingFace.contains(normalizedRepo) || normalizedRepo.contains(huggingFace)
        {
          score = max(score, 650)
        }
      }
      if score > 0 {
        scored.append((score, specification))
      }
    }
    return
      scored
      .sorted {
        if $0.score != $1.score {
          return $0.score > $1.score
        }
        let nameOrder = $0.specification.name.localizedCaseInsensitiveCompare($1.specification.name)
        if nameOrder != .orderedSame {
          return nameOrder == .orderedAscending
        }
        return $0.specification.file.localizedCaseInsensitiveCompare($1.specification.file)
          == .orderedAscending
      }
      .prefix(max(0, limit))
      .map(\.specification)
  }

  private static func sanitizeHuggingFaceRepo(_ repo: String) -> String? {
    var cleaned = repo.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    if cleaned.hasSuffix(".git") {
      cleaned = String(cleaned.dropLast(4))
    }
    let parts = cleaned.split(separator: "/").map(String.init)
    guard parts.count >= 2 else { return nil }
    return "\(parts[0])/\(parts[1])"
  }

  private static func canonicalHuggingFaceRepo(_ input: String) -> String? {
    return normalizeHuggingFaceRepo(input)?.lowercased()
  }

  private static func huggingFaceURLLike(_ input: String) -> Bool {
    guard let url = URL(string: input), let host = url.host?.lowercased() else { return false }
    return host.contains("huggingface.co")
  }

  private static func valueForCanonicalizedRepoKey<T>(_ mapping: [String: T], key: String) -> T? {
    if let value = mapping[key] {
      return value
    }
    return mapping.first { canonicalHuggingFaceRepo($0.key) == key }?.value
  }

  private static func canonicalizedRepoForLookup(_ repo: String) -> String? {
    guard let canonicalRepo = canonicalHuggingFaceRepo(repo) else { return nil }
    if let alias = valueForCanonicalizedRepoKey(huggingFaceRepoAliases, key: canonicalRepo),
      let canonicalAlias = canonicalHuggingFaceRepo(alias)
    {
      return canonicalAlias
    }
    return canonicalRepo
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
    guard specification.remoteApiModelConfig == nil else {
      return []
    }
    let model = specification.file
    let name = specification.name
    let version = ModelZoo.humanReadableNameForVersion(specification.version)
    var models: [(name: String, subtitle: String, file: String, sha256: String?)] = [
      (name: name, subtitle: version, file: model, sha256: nil)
    ]
    let textEncoder =
      specification.textEncoder
      ?? (specification.version == .v1 ? "clip_vit_l14_f16.ckpt" : "open_clip_vit_h14_f16.ckpt")
    let autoencoder = specification.autoencoder ?? "vae_ft_mse_840000_f16.ckpt"
    models.append((name: name, subtitle: version, file: textEncoder, sha256: nil))
    models.append((name: name, subtitle: version, file: autoencoder, sha256: nil))
    if let imageEncoder = specification.imageEncoder {
      models.append((name: name, subtitle: version, file: imageEncoder, sha256: nil))
    }
    if let CLIPEncoder = specification.clipEncoder {
      models.append((name: name, subtitle: version, file: CLIPEncoder, sha256: nil))
    }
    specification.additionalClipEncoders?.forEach {
      models.append((name: name, subtitle: version, file: $0, sha256: nil))
    }
    if let t5Encoder = specification.t5Encoder {
      models.append((name: name, subtitle: version, file: t5Encoder, sha256: nil))
    }
    if let diffusionMapping = specification.diffusionMapping {
      models.append((name: name, subtitle: version, file: diffusionMapping, sha256: nil))
    }
    for stageModel in (specification.stageModels ?? []) {
      models.append((name: name, subtitle: version, file: stageModel, sha256: nil))
    }
    specification.latentsUpscalers?.forEach {
      models.append((name: name, subtitle: version, file: $0.file, sha256: nil))
    }
    if let defaultRefiner = specification.defaultRefiner {
      let name = ModelZoo.humanReadableNameForModel(defaultRefiner)
      let version = ModelZoo.humanReadableNameForVersion(ModelZoo.versionForModel(defaultRefiner))
      models.append((name: name, subtitle: version, file: defaultRefiner, sha256: nil))
      if let textEncoder = ModelZoo.textEncoderForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: textEncoder, sha256: nil))
      }
      if let autoencoder = ModelZoo.autoencoderForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: autoencoder, sha256: nil))
      }
      if let imageEncoder = ModelZoo.imageEncoderForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: imageEncoder, sha256: nil))
      }
      ModelZoo.CLIPEncodersForModel(defaultRefiner).forEach { CLIPEncoder in
        models.append((name: name, subtitle: version, file: CLIPEncoder, sha256: nil))
      }
      if let T5Encoder = ModelZoo.T5EncoderForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: T5Encoder, sha256: nil))
      }
      if let diffusionMapping = ModelZoo.diffusionMappingForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: diffusionMapping, sha256: nil))
      }
      for stageModel in ModelZoo.stageModelsForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: stageModel, sha256: nil))
      }
      ModelZoo.latentsUpscalersForModel(defaultRefiner).forEach {
        models.append((name: name, subtitle: version, file: $0.file, sha256: nil))
      }
    }
    return models
  }

  /// Returns the stable, Xcode-reinstall-safe directory for Draw Things model files.
  /// Uses Application Support/Models which is never wiped on device reinstall or Xcode redeploy.
  /// Falls back to Documents/Models only if Application Support is unavailable.
  public static func persistentModelsDirectory() -> URL {
    let fileManager = FileManager.default
    // Prefer Application Support – survives Xcode reinstall on physical devices
    if let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
      let dir = appSupport.appendingPathComponent("Models", isDirectory: true)
      try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
      // One-time migration: move completed files from old Documents/Models to AppSupport/Models.
      // Guarded by UserDefaults so it only runs once, not on every path lookup.
      let migrationKey = "ModelZoo.migratedToAppSupport.v1"
      if !UserDefaults.standard.bool(forKey: migrationKey) {
        UserDefaults.standard.set(true, forKey: migrationKey)
        if let docsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
          let legacyDir = docsDir.appendingPathComponent("Models", isDirectory: true)
          if fileManager.fileExists(atPath: legacyDir.path),
             let legacyFiles = try? fileManager.contentsOfDirectory(atPath: legacyDir.path) {
            for file in legacyFiles {
              let src = legacyDir.appendingPathComponent(file)
              let dst = dir.appendingPathComponent(file)
              if !fileManager.fileExists(atPath: dst.path) {
                try? fileManager.moveItem(at: src, to: dst)
              } else {
                try? fileManager.removeItem(at: src)
              }
            }
            // Remove legacy dir if now empty
            if (try? fileManager.contentsOfDirectory(atPath: legacyDir.path))?.isEmpty == true {
              try? fileManager.removeItem(at: legacyDir)
            }
          }
        }
      }
      return dir
    }
    // Fallback
    let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
    let dir = docs.appendingPathComponent("Models", isDirectory: true)
    try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  /// Returns the legacy Documents/Models directory (used by older builds).
  /// Only used for migration and fallback lookups.
  private static func legacyDocumentsModelsDirectory() -> URL? {
    return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
      .appendingPathComponent("Models", isDirectory: true)
  }

  public static func internalFilePathForModelDownloaded(_ name: String) -> String {
    let appSupportPath = persistentModelsDirectory().appendingPathComponent(name).path
    // If the file doesn't exist in AppSupport yet, also check the legacy Documents path
    // so that ResumableDownloader can resume partial downloads from the old location.
    if !FileManager.default.fileExists(atPath: appSupportPath),
       let legacyDir = legacyDocumentsModelsDirectory() {
      let legacyPath = legacyDir.appendingPathComponent(name).path
      if FileManager.default.fileExists(atPath: legacyPath) {
        return legacyPath
      }
    }
    return appSupportPath
  }

  public static func filePathForModelDownloaded(_ name: String) -> String {
    guard let externalUrl = externalUrls.first else {
      return internalFilePathForModelDownloaded(name)
    }
    let otherFilePath = internalFilePathForModelDownloaded(name)
    let fileManager = FileManager.default
    if !isExternalUrlsPreferred && fileManager.fileExists(atPath: otherFilePath) {
      return otherFilePath
    }
    for externalUrl in externalUrls {
      let filePath = externalUrl.appendingPathComponent(name).path
      if fileManager.fileExists(atPath: filePath) {
        return filePath
      }
    }
    if fileManager.fileExists(atPath: otherFilePath) {
      return otherFilePath
    }
    return externalUrl.appendingPathComponent(name).path
  }

  public static func isModelDownloaded(
    _ specification: Specification, memorizedBy: Set<String> = []
  ) -> Bool {
    if let _ = specification.remoteApiModelConfig {
      return true
    }
    var result =
      isModelDownloaded(specification.file, memorizedBy: memorizedBy)
      && isModelDownloaded(
        specification.autoencoder ?? "vae_ft_mse_840000_f16.ckpt", memorizedBy: memorizedBy)
      && isModelDownloaded(
        specification.textEncoder
          ?? (specification.version == .v1
            ? "clip_vit_l14_f16.ckpt" : "open_clip_vit_h14_f16.ckpt"),
        memorizedBy: memorizedBy
      )
    if let imageEncoder = specification.imageEncoder {
      result = result && isModelDownloaded(imageEncoder, memorizedBy: memorizedBy)
    }
    if let clipEncoder = specification.clipEncoder {
      result = result && isModelDownloaded(clipEncoder, memorizedBy: memorizedBy)
    }
    result =
      result
      && (specification.additionalClipEncoders ?? []).allSatisfy {
        isModelDownloaded($0, memorizedBy: memorizedBy)
      }
    if let t5Encoder = specification.t5Encoder {
      result = result && isModelDownloaded(t5Encoder, memorizedBy: memorizedBy)
    }
    if let diffusionMapping = specification.diffusionMapping {
      result = result && isModelDownloaded(diffusionMapping, memorizedBy: memorizedBy)
    }
    result =
      result
      && (specification.latentsUpscalers ?? []).allSatisfy {
        isModelDownloaded($0.file, memorizedBy: memorizedBy)
      }
    return result
  }

  public static func isModelInExternalUrl(_ name: String) -> Bool {
    guard !externalUrls.isEmpty else {
      return false
    }
    for externalUrl in externalUrls {
      if FileManager.default.fileExists(atPath: externalUrl.appendingPathComponent(name).path) {
        return true
      }
    }
    return false
  }

  public static func isRemoteApiModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    guard let _ = specification.remoteApiModelConfig else {
      return false
    }
    return true
  }

  public static func isModelDownloaded(_ name: String, memorizedBy: Set<String>) -> Bool {
    guard !memorizedBy.contains(name) else {
      return true
    }
    let fileManager = FileManager.default
    for externalUrl in externalUrls {
      let filePath = externalUrl.appendingPathComponent(name).path
      if fileManager.fileExists(atPath: filePath) {
        return true
      }
    }
    let otherModelPath = internalFilePathForModelDownloaded(name)
    if fileManager.fileExists(atPath: otherModelPath) {
      return true
    }
    return false
  }

  public static func isBuiltinModel(_ name: String) -> Bool {
    return builtinModels.contains(name)
  }

  public static func humanReadableNameForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return name }
    return specification.name
  }

  public static func textPrefixForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return "" }
    return specification.prefix
  }

  public static func textEncoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.textEncoder
  }

  public static func textEncoderVersionForModel(_ name: String) -> TextEncoderVersion? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.textEncoderVersion
  }

  public static func imageEncoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.imageEncoder
  }

  public static func CLIPEncodersForModel(_ name: String) -> [String] {
    guard let specification = specificationForModel(name) else { return [] }
    return (specification.clipEncoder.map { [$0] } ?? [])
      + (specification.additionalClipEncoders ?? [])
  }

  public static func T5EncoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.t5Encoder
  }

  public static func diffusionMappingForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.diffusionMapping
  }

  public static func versionForModel(_ name: String) -> ModelVersion {
    guard let specification = specificationForModel(name) else { return .v1 }
    return specification.version
  }

  public static func noteForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return "" }
    return specification.note ?? ""
  }

  public static func copyrightForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return "" }
    return specification.copyright ?? ""
  }

  public static func autoencoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.autoencoder
  }

  public static func latentsUpscalersForModel(_ name: String) -> [Specification.LatentsUpscaler] {
    guard let specification = specificationForModel(name) else { return [] }
    return specification.latentsUpscalers ?? []
  }

  public static func builtinLoRAForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.builtinLora ?? false
  }

  public static func isModelDeprecated(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.deprecated ?? false
  }

  public static func objectiveForModel(_ name: String) -> Denoiser.Objective {
    guard let specification = specificationForModel(name) else { return .epsilon }
    if let objective = specification.objective {
      return objective
    }
    if specification.predictV == true {
      return .v
    }
    switch specification.version {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .pixart:
      return .epsilon
    case .cosmos2_5_2b:
      return .u(conditionScale: 1)
    case .sd3, .sd3Large, .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1,
      .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage, .flux2, .flux2_9b, .flux2_4b,
      .ltx2, .ltx2_3, .ernieImage, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2,
      .longcatVideoAvatar1_5, .minimaxH3:
      return .u(conditionScale: 1000)
    }
  }

  public static func conditioningForModel(_ name: String) -> Denoiser.Conditioning {
    guard let specification = specificationForModel(name) else { return .timestep }
    if let conditioning = specification.conditioning {
      return conditioning
    }
    switch specification.version {
    case .kandinsky21, .sdxlBase, .sdxlRefiner, .v1, .v2, .ssd1b, .wurstchenStageC,
      .wurstchenStageB, .sd3, .sd3Large, .pixart, .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b,
      .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
      .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ltx2, .ltx2_3, .seedvr2_3b,
      .seedvr2_7b, .ideogram4, .krea2, .longcatVideoAvatar1_5, .minimaxH3:
      return .timestep
    case .svdI2v:
      return .noise
    }
  }

  public static func noiseDiscretizationForModel(_ name: String) -> NoiseDiscretization {
    guard let specification = specificationForModel(name) else {
      return .ddpm(
        .init(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, linspace: .linearWrtSigma))
    }
    if let noiseDiscretization = specification.noiseDiscretization {
      return noiseDiscretization
    }
    switch specification.version {
    case .kandinsky21:
      return .ddpm(
        .init(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, linspace: .linearWrtBeta))
    case .pixart:
      return .ddpm(
        .init(linearStart: 0.0001, linearEnd: 0.02, timesteps: 1_000, linspace: .linearWrtBeta))
    case .sdxlBase, .sdxlRefiner, .ssd1b, .v1, .v2:
      return .ddpm(
        .init(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, linspace: .linearWrtSigma))
    case .svdI2v:
      return .edm(.init(sigmaMax: 700.0))
    case .wurstchenStageC, .wurstchenStageB:
      return .edm(.init(sigmaMin: 0.01, sigmaMax: 99.995))
    case .qwenImage2_1:
      return .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000, shiftTerminal: 0.02))
    case .cosmos2_5_2b:
      return .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1))
    case .sd3, .sd3Large, .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1,
      .hiDreamO1, .qwenImage, .wan22_5b, .zImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3,
      .ernieImage, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2,
      .longcatVideoAvatar1_5, .minimaxH3:
      return .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
    }
  }

  public static func paddedTextEncodingLengthForModel(_ name: String) -> Int {
    guard let specification = specificationForModel(name) else { return 0 }
    if let paddedTextEncodingLength = specification.paddedTextEncodingLength {
      return paddedTextEncodingLength
    }
    switch specification.version {
    case .v1, .v2, .svdI2v, .ssd1b, .sdxlBase, .sdxlRefiner, .kandinsky21, .wurstchenStageB,
      .wurstchenStageC, .sd3, .sd3Large:
      return 77
    case .pixart, .minimaxH3:
      return 0
    case .auraflow:
      return 256
    case .flux1:
      return 512
    case .hiDreamI1:
      return 128
    case .hiDreamO1, .hunyuanVideo, .qwenImage, .qwenImage2_1, .zImage, .ernieImage, .flux2, .ltx2,
      .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
      return 0
    case .wan21_1_3b, .wan21_14b, .wan22_5b, .flux2_9b, .flux2_4b, .cosmos2_5_2b,
      .longcatVideoAvatar1_5:
      return 512
    }
  }

  public static func latentsScalingForModel(_ name: String) -> (
    mean: [Float]?, std: [Float]?, scalingFactor: Float, shiftFactor: Float?, audioMean: [Float]?,
    audioStd: [Float]?
  ) {
    guard let specification = specificationForModel(name) else {
      return (nil, nil, 1, nil, nil, nil)
    }
    let audioLatentsMean = specification.audioLatentsMean
    let audioLatentsStd = specification.audioLatentsStd
    if let mean = specification.latentsMean, let std = specification.latentsStd,
      let scalingFactor = specification.latentsScalingFactor
    {
      return (mean, std, scalingFactor, nil, audioLatentsMean, audioLatentsStd)
    }
    if let scalingFactor = specification.latentsScalingFactor {
      return (nil, nil, scalingFactor, nil, audioLatentsMean, audioLatentsStd)
    }
    switch specification.version {
    case .v1, .v2, .svdI2v:
      return (nil, nil, 0.18215, nil, nil, nil)
    case .ssd1b, .sdxlBase, .sdxlRefiner, .pixart, .auraflow:
      return (nil, nil, 0.13025, nil, nil, nil)
    case .kandinsky21, .hiDreamO1:
      return (nil, nil, 1, nil, nil, nil)
    case .minimaxH3:
      return (
        [
          0.85809034, -0.96065915, 1.066164, -0.50903255, -0.2727582, -1.3675414,
          -0.2553255, -0.26907554, -0.5376841, -0.04640973, 0.66573703, 0.19690128,
          -0.5460608, -0.4035342, -0.23683025, 0.25928453, -0.30133945, 0.21134199,
          -1.1206849, 0.35819334, -0.042251438, 0.260483, 0.22864093, 0.7056032,
        ],
        [
          1.2223774, 1.2767264, 1.6831775, 1.7549455, 1.5636216, 2.1941435,
          0.9653138, 1.0569886, 0.8419489, 0.7729953, 1.8955938, 0.94684184,
          0.79968095, 0.449889, 0.71974, 0.6936293, 2.961095, 2.76942,
          3.0496185, 2.1088054, 3.2762263, 3.1627357, 2.2816813, 2.6127844,
        ], 1, nil,
        [
          -0.020211687, 0.38764665, -0.0439828, -0.28591514, 0.08179686, -0.3578264,
          0.04062381, -0.01552535, -0.22336248, 0.18210068, 0.2941779, -0.07901168,
          -0.056815073, -0.36990282, -0.31616315, 0.5905951, -0.05213957, 0.01367316,
          -0.03691648, 0.09732661, -0.33946624, -0.30685678, -0.24504599, -0.03469852,
          0.02868032, -0.2121778, -0.16782632, 0.3221288, -0.12230559, 0.43566048,
          -0.05025992, 0.39792585,
        ],
        [
          1.6895524, 2.7626374, 1.7945344, 1.6801682, 1.6390227, 2.7788298, 1.765909,
          1.6199758, 2.6336527, 1.8539357, 2.5056498, 1.8110192, 1.9579657, 1.6685498,
          1.4922469, 3.2986703, 1.9491805, 1.8720003, 1.833408, 1.648807, 1.6176958,
          1.9131449, 1.5695245, 1.6943659, 1.8318421, 1.5540638, 1.9344931, 1.5991982,
          1.718046, 1.630722, 1.8661226, 1.5613768,
        ]
      )
    case .wurstchenStageC, .wurstchenStageB:
      return (nil, nil, 2.32558139535, nil, nil, nil)
    case .sd3, .sd3Large:
      return (nil, nil, 1.5305, 0.0609, nil, nil)
    case .flux1, .hiDreamI1, .zImage:
      return (nil, nil, 0.3611, 0.11590, nil, nil)
    case .seedvr2_3b, .seedvr2_7b:
      return (nil, nil, 0.9152, nil, nil, nil)
    case .hunyuanVideo:
      return (nil, nil, 0.476986, nil, nil, nil)
    case .qwenImage2_1:
      return (
        [
          0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177,
          -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954,
          -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036,
          -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914,
          -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326,
          -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121,
          -0.1562, 0.0304,
        ],
        [
          3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308,
          3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454,
          3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037,
          3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819,
          3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408,
          3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
        ], 1, nil, nil, nil
      )
    case .wan21_1_3b, .wan21_14b, .qwenImage, .cosmos2_5_2b, .krea2,
      .longcatVideoAvatar1_5:
      return (
        [
          -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
          0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ],
        [
          2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
          3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ], 1, nil, nil, nil
      )
    case .wan22_5b:
      return (
        [
          -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542,
          0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109,
          0.2665, -0.2108, -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899,
          -0.2799, -0.1230, -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
          0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
        ],
        [
          0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894,
          1.0901, 0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146,
          0.8986, 0.5659, 0.7069, 0.5338, 0.4889, 0.4917, 0.4069,
          0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
          0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ], 1, nil, nil, nil
      )
    case .ideogram4:
      return (
        [
          0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549,
          0.05021099, -0.15083604, -0.15360136, -0.20131799, 0.01922352, 0.0622626,
          0.10140969, -0.06739428, 0.3758261, -0.233712, 0.35164491, -0.02590912,
          -0.0271935, -0.10833897, -0.1476848, -0.01130957, -0.2298372, 0.23526423,
          -0.10893522, 0.11957631, 0.04047799, 0.3134589, -0.17225064, -0.18646109,
          -0.34691978, -0.03571246, 0.02583857, 0.10190072, 0.28402294, 0.26952152,
          -0.21634675, -0.17938656, 0.04358909, -0.15007621, -0.1548502, -0.18971131,
          0.02710861, 0.05609494, 0.10697846, -0.06854968, 0.38167698, -0.24269937,
          0.35705471, -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137,
          -0.21863696, 0.23228983, -0.11739769, 0.11693044, 0.02563311, 0.31356594,
          -0.17420591, -0.19006285, -0.34905377, -0.04025005, 0.01924137, 0.07652984,
          0.2995608, 0.2628057, -0.22011674, -0.12715361, 0.04879879, -0.14075719,
          -0.15935895, -0.2123584, 0.01974813, 0.05523547, 0.10011992, -0.06428964,
          0.37781868, -0.21491644, 0.34254215, -0.03153528, -0.0310082, -0.10761415,
          -0.14730405, -0.02475182, -0.2285588, 0.2515081, -0.10445128, 0.12446,
          0.07062869, 0.30880162, -0.18016875, -0.18869164, -0.34533499, -0.0129177,
          0.02578168, 0.07993659, 0.28642181, 0.26038408, -0.22459419, -0.14820155,
          0.04059549, -0.14043529, -0.16111187, -0.2020305, 0.02602069, 0.04852717,
          0.10432153, -0.06309942, 0.38402443, -0.22397003, 0.34814481, -0.03774432,
          -0.03381438, -0.11245691, -0.14128767, -0.02853208, -0.21752016, 0.24872463,
          -0.11399775, 0.1222687, 0.05620835, 0.309178, -0.18065738, -0.19401479,
          -0.34495114, -0.01760592,
        ],
        [
          1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.6675316, 1.69059584,
          1.56853198, 1.62314944, 1.89106626, 1.58086668, 1.60822129, 1.60962993,
          1.63322129, 1.56074359, 1.73419528, 1.7919265, 1.64040632, 1.66802808,
          1.60390303, 1.75480492, 1.63187587, 1.64334594, 1.61722884, 1.60146046,
          1.63459219, 1.55291476, 1.68771497, 1.68415657, 1.78966054, 1.66631641,
          1.65626686, 1.65976433, 1.63487607, 1.69513249, 1.72933756, 1.91310663,
          1.67035057, 1.72286863, 1.56719251, 1.61934825, 1.88628859, 1.56911539,
          1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003, 1.77563606,
          1.63732541, 1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667,
          1.61659342, 1.59722044, 1.64103121, 1.5408531, 1.68610394, 1.67772755,
          1.78998563, 1.66621713, 1.65458955, 1.66041308, 1.64710857, 1.68163503,
          1.74000294, 1.92784786, 1.67411194, 1.67395548, 1.57406532, 1.62199356,
          1.87618195, 1.5584375, 1.57438785, 1.61711053, 1.63094305, 1.55644029,
          1.73124302, 1.80666627, 1.6463621, 1.65932006, 1.60816188, 1.75682671,
          1.64695873, 1.63121722, 1.61380832, 1.60478651, 1.63396035, 1.53505068,
          1.65534289, 1.67132281, 1.80317197, 1.6767314, 1.65700938, 1.68426259,
          1.65339716, 1.67540638, 1.73298504, 1.94067348, 1.67893609, 1.70635117,
          1.5730906, 1.61928553, 1.87148809, 1.56244866, 1.56697152, 1.61584394,
          1.62759496, 1.55480378, 1.73484107, 1.79055143, 1.64688773, 1.66121492,
          1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592, 1.60792883,
          1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476,
          1.65655173, 1.68533454,
        ], 1, nil, nil, nil
      )
    case .ernieImage, .flux2, .flux2_9b, .flux2_4b:
      return (
        [
          -6.76177666e-02, -7.15223551e-02, -7.53413364e-02, -7.44939372e-02,
          2.22785398e-02, 1.79953799e-02, 1.41973700e-02, 1.83613356e-02,
          -6.27551854e-05, -6.25144318e-03, -2.10153405e-04, -3.13947396e-03,
          -2.72027273e-02, -2.81060152e-02, -2.76455786e-02, -2.90332772e-02,
          -7.68895298e-02, -6.71701953e-02, -9.01882946e-02, -8.92138183e-02,
          1.68366600e-02, 1.52064804e-02, 7.90204294e-03, 8.57926160e-03,
          8.34798254e-03, 1.54090952e-03, 2.58349784e-04, -4.28175228e-03,
          -4.38771434e-02, -4.18955982e-02, -4.37803492e-02, -4.31488380e-02,
          -1.02466689e-02, -1.31864231e-02, -6.62019709e-03, -4.76623932e-03,
          -3.10628936e-02, -3.05543691e-02, -2.79040541e-02, -1.79539975e-02,
          3.02119297e-03, 1.50253996e-03, 1.25925653e-02, 1.44742327e-02,
          3.47208753e-02, 3.37658636e-02, 3.36632989e-02, 2.82952897e-02,
          1.97971705e-03, 4.72892029e-03, 4.65414440e-03, 4.96361824e-03,
          1.22726467e-02, 8.09616689e-03, 8.05679616e-03, 1.45769194e-02,
          6.81073293e-02, 6.79029524e-02, 7.66535401e-02, 7.31865391e-02,
          -4.62144315e-02, -4.73941378e-02, -3.91875766e-02, -5.10934070e-02,
          -5.27758673e-02, -4.77382541e-02, -4.70039584e-02, -5.17151840e-02,
          -3.17052379e-02, -3.16338688e-02, -3.44672315e-02, -2.82559078e-02,
          5.09686768e-02, 4.45049144e-02, 5.78130186e-02, 4.58035618e-02,
          -4.11602221e-02, -4.58290428e-02, -4.87412103e-02, -4.67392765e-02,
          -8.83873831e-03, -1.06276469e-02, -8.80550127e-03, -4.61349264e-03,
          -3.75848413e-02, -4.32198308e-02, -4.35743667e-02, -4.98905331e-02,
          1.18464455e-02, 1.66369155e-02, 2.02845689e-02, 2.78996639e-02,
          1.12712244e-02, 1.29012959e-02, 1.55935134e-03, 7.15561956e-03,
          -1.18002137e-02, -1.83626905e-03, -1.41415279e-02, -5.37070679e-03,
          -9.09713656e-03, -1.37955081e-02, -1.44679286e-02, -1.86988190e-02,
          3.22541557e-02, 3.05014588e-02, 2.58702636e-02, 2.99565904e-02,
          5.39954007e-02, 6.14439026e-02, 4.95390743e-02, 5.89892939e-02,
          -5.10806963e-02, -6.03262000e-02, -4.77751829e-02, -5.23972921e-02,
          -2.26762425e-02, -2.74192505e-02, -1.53651498e-02, -2.54624709e-02,
          -5.72077744e-02, -5.64766899e-02, -5.17635308e-02, -4.95564640e-02,
          1.15854675e-02, 5.42225968e-03, 1.63003802e-02, 1.03847245e-02,
        ],
        [
          1.8028622, 1.778625, 1.7867663, 1.7836679, 1.7717245, 1.7590446, 1.761046,
          1.7479103, 1.733573, 1.7372522, 1.7340292, 1.7342823, 1.8625563, 1.8527341,
          1.8628826, 1.8588936, 1.7592851, 1.7525632, 1.7555695, 1.7582504, 1.7363207,
          1.7399888, 1.7355365, 1.7393657, 1.7342308, 1.7245967, 1.7392466, 1.7303607,
          1.7551407, 1.7512879, 1.7559179, 1.7487845, 1.8449168, 1.8454357, 1.854986,
          1.8535364, 1.8239557, 1.78132, 1.7853717, 1.7945118, 1.8047395, 1.7875748,
          1.7694789, 1.7675722, 1.7782369, 1.7666509, 1.7925009, 1.78479, 1.7579452,
          1.7407266, 1.7483361, 1.7367632, 1.796092, 1.7997854, 1.7919755, 1.7925386,
          1.7779572, 1.7747114, 1.772703, 1.7749013, 1.7526383, 1.7447339, 1.7657429,
          1.7495404, 1.7774895, 1.7720175, 1.7813174, 1.781325, 1.8161929, 1.8012798,
          1.8023334, 1.8032819, 1.7527498, 1.7331196, 1.7562932, 1.7482262, 1.7609811,
          1.7507315, 1.768117, 1.7612798, 1.7665246, 1.7545255, 1.7827506, 1.7725668,
          1.7895597, 1.7999352, 1.7863562, 1.7759794, 1.7613311, 1.7625202, 1.7559929,
          1.7576683, 1.7782613, 1.7671335, 1.7810351, 1.7799498, 1.7201267, 1.7067566,
          1.7265412, 1.7091043, 1.7792666, 1.7577881, 1.750232, 1.7454648, 1.7586769,
          1.7499638, 1.7524506, 1.7361721, 1.761603, 1.7572367, 1.7444118, 1.7430167,
          1.7508849, 1.760951, 1.7634183, 1.7612051, 1.7253897, 1.71346, 1.7320756,
          1.7226279, 1.7664, 1.7623769, 1.771846, 1.7664031, 1.7457305, 1.7441442,
          1.756882, 1.7530031,
        ], 1, nil, nil, nil
      )
    case .ltx2:
      return (
        [
          -0.07763672, -0.06201172, 0.07324219, 0.08935547, 0.00294495, 0.00245667,
          0.01342773, -0.05419922, 0.02246094, 0.03735352, -0.02575684, 0.05517578,
          -0.05029297, 0.01721191, 0.00994873, 0.04223633, -0.03271484, 0.0559082,
          -0.04858398, -0.00598145, -0.00854492, 0.00891113, -0.00671387, 0.01074219,
          -0.04003906, 0.03076172, -0.00349426, 0.01977539, -0.05810547, -0.0300293,
          -0.04931641, -0.04077148, 0.01153564, -0.00933838, 0.0291748, 0.00454712,
          -0.01324463, 0.02954102, -0.01989746, 0.02612305, -0.06445312, 0.01855469,
          -0.10009766, -0.03271484, 0.05249023, 0.125, -0.01049805, 0.03149414,
          0.16601562, -0.04443359, -0.01635742, 0.01245117, 0.01239014, -0.03442383,
          -0.01672363, 0.04199219, -0.00318909, -0.00524902, -0.04052734, -0.01806641,
          -0.05810547, 0.02087402, 0.03466797, 0.00242615, -0.09375, -0.05053711,
          0.07421875, -0.04101562, -0.07714844, 0.00189972, -0.09716797, -0.00463867,
          -0.00585938, 0.00674438, 0.04077148, -0.00341797, 0.06201172, 0.08300781,
          0.09277344, -0.00370789, 0.03149414, -0.04882812, -0.02062988, -0.00512695,
          -0.05493164, -0.06835938, 0.00068665, -0.0324707, -0.01867676, 0.01019287,
          0.00570679, -0.04223633, 0.109375, 0.0133667, -0.04150391, -0.00939941,
          -0.05029297, 0.01904297, 0.04052734, 0.03637695, -0.00512695, 0.00933838,
          -0.00205994, 0.00680542, 0.02331543, -0.00872803, 0.01098633, -0.015625,
          -0.15429688, -0.00302124, -0.01257324, -0.03027344, 0.01794434, 0.00524902,
          0.03955078, -0.09619141, -0.03149414, 0.0090332, -0.00201416, -0.03979492,
          -0.04711914, -0.0035553, -0.01647949, -0.09082031, 0.00799561, -0.06494141,
          0.14257812, 0.02331543,
        ],
        [
          0.16113281, 0.12792969, 0.16894531, 0.18945312, 0.13574219, 0.12207031,
          0.15039062, 0.19335938, 0.09912109, 0.12011719, 0.12402344, 0.11767578,
          0.16015625, 0.24414062, 0.14355469, 0.13671875, 0.11328125, 0.12890625,
          0.22265625, 0.15039062, 0.12060547, 0.09277344, 0.296875, 0.12792969,
          0.14160156, 0.1796875, 0.14453125, 0.125, 0.14453125, 0.09619141,
          0.19140625, 0.11425781, 0.17871094, 0.13183594, 0.10351562, 0.09960938,
          0.12988281, 0.16699219, 0.10546875, 0.10302734, 0.12695312, 0.10302734,
          0.20507812, 0.23925781, 0.16796875, 0.1875, 0.10449219, 0.25390625,
          0.17089844, 0.12988281, 0.15820312, 0.09912109, 0.1171875, 0.14746094,
          0.12988281, 0.20996094, 0.14160156, 0.12451172, 0.15332031, 0.1328125,
          0.11669922, 0.140625, 0.22070312, 0.10888672, 0.3515625, 0.11572266,
          0.15332031, 0.11425781, 0.21777344, 0.12158203, 0.14453125, 0.16796875,
          0.19824219, 0.13476562, 0.13867188, 0.12060547, 0.11865234, 0.20019531,
          0.10693359, 0.12695312, 0.11621094, 0.14453125, 0.25390625, 0.11669922,
          0.18164062, 0.14160156, 0.14648438, 0.12695312, 0.10742188, 0.12109375,
          0.10644531, 0.21582031, 0.43359375, 0.10595703, 0.18652344, 0.13085938,
          0.11279297, 0.12109375, 0.11181641, 0.1328125, 0.16113281, 0.10839844,
          0.15332031, 0.12792969, 0.12109375, 0.10449219, 0.10742188, 0.09667969,
          0.34960938, 0.10302734, 0.10888672, 0.20898438, 0.15039062, 0.22851562,
          0.11279297, 0.15917969, 0.15625, 0.11035156, 0.10498047, 0.13085938,
          0.09521484, 0.11279297, 0.13769531, 0.16601562, 0.15039062, 0.12597656, 0.27929688,
          0.10742188,
        ], 1, nil,
        [
          1.5390625, 2.84375, 2.359375, 1.9921875, 1.5390625, 1.296875,
          1.0859375, 0.87109375, 0.62890625, 0.40429688, 0.19726562, -0.10205078,
          -0.43164062, -0.734375, -1.0546875, -0.65234375, 0.4375, -0.6796875,
          -0.4296875, -0.296875, -0.11279297, 0.03564453, 0.14550781, 0.21191406,
          0.29492188, 0.38867188, 0.46289062, 0.57421875, 0.66015625, 0.89453125,
          0.49023438, 0.70703125, 0.75390625, 0.41210938, 0.36523438, 0.14941406,
          -0.00799561, -0.12255859, -0.19335938, -0.31054688, -0.43164062, -0.48632812,
          -0.64453125, -0.73828125, -0.87890625, -1.0703125, -0.94140625, -0.9765625,
          0.06445312, -0.37304688, -0.13867188, -0.2578125, -0.38476562, -0.49804688,
          -0.5625, -0.59375, -0.69140625, -0.765625, -0.78515625, -0.87890625,
          -0.8984375, -0.9765625, -1.0859375, -0.5078125, 1.6171875, 1.671875,
          1.7890625, 1.7421875, 1.7109375, 1.6875, 1.6640625, 1.609375,
          1.546875, 1.546875, 1.5078125, 1.4296875, 1.390625, 1.203125,
          1.3125, 0.7734375, -0.2421875, -1.9609375, -1.328125, -1.203125,
          -0.9921875, -0.796875, -0.703125, -0.57421875, -0.46679688, -0.36328125,
          -0.22753906, -0.10107422, 0.01660156, 0.23144531, 0.375, 0.15625,
          0.9609375, 0.5234375, 0.58984375, 0.3125, 0.1640625, 0.06787109,
          0.00631714, -0.09326172, -0.17480469, -0.22460938, -0.31640625, -0.37695312,
          -0.47460938, -0.703125, -0.82421875, -1.484375, 0.21875, 0.11181641,
          0.2890625, 0.00704956, -0.234375, -0.40820312, -0.52734375, -0.671875,
          -0.7890625, -0.94921875, -1.0703125, -1.2578125, -1.390625, -1.5859375,
          -1.671875, -1.804687,
        ],
        [
          1.9921875, 2.125, 2.078125, 2.046875, 2.046875, 2.03125,
          2.015625, 1.9765625, 1.984375, 1.96875, 1.96875, 1.9453125,
          1.9296875, 1.90625, 1.9453125, 1.7265625, 1.09375, 1.265625,
          1.2421875, 1.203125, 1.171875, 1.1484375, 1.1171875, 1.0859375,
          1.0859375, 1.0546875, 1.0234375, 1.0078125, 0.98046875, 0.96484375,
          0.90625, 0.8828125, 1.3125, 1.328125, 1.2890625, 1.234375,
          1.21875, 1.1796875, 1.1796875, 1.1171875, 1.0625, 1.0390625,
          1.046875, 1.015625, 0.99609375, 0.98046875, 1.0078125, 1.015625,
          1.4296875, 1.2109375, 1.1484375, 1.1015625, 1.078125, 1.0390625,
          0.98828125, 0.98046875, 0.94921875, 0.90234375, 0.90234375, 0.875,
          0.8671875, 0.84765625, 0.8203125, 0.9140625, 0.76953125, 1.109375,
          1.109375, 1.09375, 1.078125, 1.0703125, 1.0390625, 1.0078125,
          0.9609375, 0.9140625, 0.87109375, 0.82421875, 0.79296875, 0.77734375,
          0.7734375, 0.75390625, 1.3203125, 1.3828125, 1.296875, 1.296875,
          1.2578125, 1.2421875, 1.234375, 1.203125, 1.1796875, 1.171875,
          1.1796875, 1.125, 1.125, 1.1015625, 1.109375, 1.171875,
          1.46875, 1.28125, 1.2421875, 1.2109375, 1.2109375, 1.1875,
          1.140625, 1.1328125, 1.109375, 1.09375, 1.0859375, 1.0625,
          1.0546875, 1.0625, 1.1015625, 1.3515625, 1.203125, 1.5625,
          1.4609375, 1.4453125, 1.3984375, 1.390625, 1.359375, 1.328125,
          1.3125, 1.2890625, 1.296875, 1.2421875, 1.234375, 1.21875,
          1.2265625, 1.054687,
        ]
      )
    case .ltx2_3:
      return (
        [
          0.0222167969, 0.00282287598, -0.0121459961, -0.0131835938, 0.0137939453, -0.0227050781,
          -0.0127563477, -0.0111694336, 0.0141601562, -0.0119628906, -7.77244568e-05,
          -0.00127410889, 0.0122070312, -0.0502929688, 0.000356674194, 0.00263977051,
          0.00643920898, 0.00576782227, 0.0041809082, 0.00634765625, -0.0269775391, 0.00653076172,
          0.00318908691, 0.0361328125, -0.0581054688, -0.00245666504,
          0.0451660156, -0.00140380859, 0.0174560547, -0.00194549561, -0.00982666016, 0.00567626953,
          0.000247955322, -0.000820159912, 0.00521850586, -0.0103759766,
          0.0065612793, 0.0116577148, -0.00842285156, 0.000984191895, 0.00433349609, 0.0515136719,
          0.000185012817, -0.0113525391, 0.00823974609, -0.330078125,
          -0.00561523438, 0.00982666016, -0.00677490234, 0.00118255615, -0.0173339844,
          -0.0164794922, -0.00198364258, -0.00147247314, 0.00205993652, 0.0190429688,
          0.00848388672, -0.108886719, -0.00805664062, 0.00393676758, -0.00915527344,
          0.000926971436, 0.0137329102, 0.166015625, -0.010559082, 0.0275878906,
          0.000881195068, 0.00775146484, 0.0346679688, -0.00157928467, 0.0030670166, -0.00207519531,
          -0.102050781, 0.00118255615, 0.0116577148, 0.427734375,
          0.00137329102, 0.00473022461, -0.00653076172, -0.00811767578, -0.0229492188,
          -0.0131835938, 0.00415039062, 0.056640625, 0.00215148926, 0.00219726562,
          -0.0981445312, -0.0145263672, -0.00361633301, -0.00482177734, 0.00204467773, -0.0234375,
          -0.0498046875, 0.00598144531, -0.00643920898, 0.0020904541,
          0.00686645508, 0.229492188, -0.0446777344, -0.00738525391, 0.00451660156, -0.0118408203,
          0.00109863281, -0.5546875, 0.0136108398, -0.0314941406,
          0.00267028809, -0.00970458984, 0.115722656, -0.0654296875, -0.00674438477, -0.0104980469,
          0.00201416016, -0.000820159912, -0.0134277344, -0.00102996826,
          0.306640625, 0.0078125, -0.00823974609, 0.0114135742, -0.00396728516, -0.0354003906,
          0.00384521484, 0.00531005859, -0.000896453857, -0.000191688538,
          0.0297851562, -0.00598144531,
        ],
        [
          0.23828125, 0.0751953125, 0.103515625, 0.0947265625, 0.123046875, 0.139648438,
          0.223632812,
          0.365234375, 0.157226562, 0.186523438, 0.0922851562, 0.140625, 0.166015625, 0.251953125,
          0.0883789062, 0.123535156, 0.138671875, 0.109863281, 0.106933594,
          0.138671875, 0.197265625, 0.103515625, 0.1328125, 0.208984375, 0.1796875, 0.147460938,
          0.162109375, 0.129882812, 0.118164062, 0.0771484375, 0.146484375,
          0.0849609375, 0.0854492188, 0.0751953125, 0.125, 0.095703125, 0.125, 0.137695312, 0.09375,
          0.0913085938, 0.0776367188, 0.177734375, 0.0844726562,
          0.0952148438, 0.124023438, 0.190429688, 0.09375, 0.0854492188, 0.15234375, 0.0932617188,
          0.141601562, 0.129882812, 0.08203125, 0.12109375, 0.100097656,
          0.193359375, 0.0947265625, 0.6953125, 0.087890625, 0.0805664062, 0.118652344,
          0.0825195312, 0.109375, 0.53125, 0.211914062, 0.225585938, 0.109375,
          0.138671875, 0.15625, 0.0966796875, 0.0952148438, 0.0942382812, 0.486328125, 0.07421875,
          0.116699219, 0.75390625, 0.124511719, 0.0903320312, 0.123535156,
          0.104980469, 0.16796875, 0.143554688, 0.165039062, 0.16796875, 0.100585938, 0.0747070312,
          0.208984375, 0.139648438, 0.100585938, 0.126953125,
          0.0942382812, 0.165039062, 0.23828125, 0.126953125, 0.141601562, 0.153320312, 0.202148438,
          0.59765625, 0.19140625, 0.087890625, 0.125976562, 0.17578125,
          0.0991210938, 0.9140625, 0.109375, 0.154296875, 0.100585938, 0.127929688, 0.322265625,
          0.208007812, 0.345703125, 0.12890625, 0.16015625, 0.150390625,
          0.186523438, 0.215820312, 0.7109375, 0.0986328125, 0.100097656, 0.120605469, 0.107421875,
          0.171875, 0.124023438, 0.107421875, 0.0844726562, 0.0942382812,
          0.182617188, 0.116699219,
        ], 1, nil,
        [
          1.34375, 2.734375, 2.296875, 1.8984375, 1.4609375, 1.203125, 0.98828125, 0.859375,
          0.6328125, 0.39453125, 0.177734375, -0.140625, -0.51171875, -0.86328125, -1.1953125,
          -0.79296875, 0.59375, -0.56640625, -0.37890625, -0.23046875,
          -0.0277099609, 0.115234375, 0.2265625, 0.2734375, 0.333984375, 0.431640625, 0.494140625,
          0.62890625, 0.7265625, 0.98828125, 0.58203125, 0.7734375, 0.75,
          0.337890625, 0.33203125, 0.116699219, -0.0358886719, -0.185546875, -0.229492188,
          -0.3046875, -0.4453125, -0.494140625, -0.65625, -0.7578125, -0.9375,
          -1.1328125, -1.0, -1.0625, -0.0678710938, -0.4140625, -0.130859375, -0.26953125, -0.375,
          -0.515625, -0.57421875, -0.58203125, -0.6484375, -0.7265625,
          -0.73046875, -0.83203125, -0.85546875, -0.9609375, -1.078125, -0.498046875, 1.6015625,
          1.6875, 1.84375, 1.7890625, 1.75, 1.703125, 1.6640625, 1.6328125,
          1.609375, 1.5625, 1.515625, 1.453125, 1.4140625, 1.2265625, 1.3046875, 0.76171875,
          -0.0942382812, -1.90625, -1.296875, -1.1875, -0.9609375, -0.7578125,
          -0.6640625, -0.5703125, -0.49609375, -0.384765625, -0.279296875, -0.135742188,
          0.0212402344, 0.25, 0.41796875, 0.200195312, 0.8359375, 0.5078125,
          0.609375, 0.333984375, 0.16015625, 0.0639648438, -0.00891113281, -0.0471191406,
          -0.10546875, -0.170898438, -0.2421875, -0.34375, -0.484375, -0.734375,
          -0.86328125, -1.546875, 0.0712890625, 0.0262451172, 0.209960938, -0.0537109375,
          -0.2890625, -0.48046875, -0.6015625, -0.7265625, -0.8125, -0.97265625,
          -1.09375, -1.28125, -1.4453125, -1.65625, -1.765625, -1.8984375,
        ],
        [
          1.875, 2.046875, 2.03125, 2.03125, 1.9765625, 1.953125, 1.9453125, 1.9140625, 1.875,
          1.8515625, 1.8359375, 1.8125, 1.796875, 1.796875, 1.859375, 1.65625, 1.0390625, 1.1796875,
          1.171875, 1.1328125, 1.1015625, 1.0625, 1.03125, 1.0, 0.96875,
          0.9296875, 0.91796875, 0.89453125, 0.875, 0.85546875, 0.80078125, 0.77734375, 1.3046875,
          1.3203125, 1.296875, 1.2265625, 1.171875, 1.125, 1.1015625,
          1.0625, 1.015625, 0.99609375, 0.98046875, 0.96875, 0.94140625, 0.9453125, 0.984375,
          1.03125, 1.3984375, 1.171875, 1.125, 1.0625, 1.015625, 0.97265625,
          0.9296875, 0.91796875, 0.8828125, 0.8359375, 0.8359375, 0.8203125, 0.80859375, 0.79296875,
          0.7734375, 0.8828125, 0.76171875, 1.078125, 1.046875,
          1.0546875, 1.03125, 1.015625, 0.98046875, 0.94921875, 0.90234375, 0.859375, 0.828125,
          0.79296875, 0.765625, 0.75, 0.75390625, 0.7421875, 1.2421875,
          1.328125, 1.25, 1.25, 1.21875, 1.1796875, 1.1796875, 1.15625, 1.125, 1.109375, 1.09375,
          1.0703125, 1.0703125, 1.046875, 1.0703125, 1.125, 1.390625,
          1.234375, 1.21875, 1.2109375, 1.15625, 1.140625, 1.109375, 1.1171875, 1.09375, 1.0703125,
          1.0625, 1.046875, 1.03125, 1.0390625, 1.09375, 1.359375,
          1.15625, 1.5234375, 1.4453125, 1.4296875, 1.3828125, 1.3359375, 1.3203125, 1.2734375,
          1.2421875, 1.2109375, 1.1875, 1.15625, 1.1484375, 1.1328125,
          1.140625, 0.99609375,
        ]
      )
    }
  }

  public static func noiseScalingFactorForModel(_ name: String) -> Float {
    guard let specification = specificationForModel(name) else { return 1 }
    return specification.noiseScalingFactor ?? 1
  }

  public static func audioEncoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    switch specification.version {
    case .longcatVideoAvatar1_5:
      return "whisper_large_v3_f16.ckpt"
    case .minimaxH3:
      return autoencoderForModel(name)
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b,
      .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
      .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .ltx2, .ltx2_3,
      .seedvr2_3b, .seedvr2_7b:
      return nil
    }
  }

  public static func audioSampleRateForModel(_ name: String) -> Int {
    guard let specification = specificationForModel(name) else {
      return 24_000
    }
    switch specification.version {
    case .v1, .v2, .svdI2v, .ssd1b, .sdxlBase, .sdxlRefiner, .pixart, .auraflow, .kandinsky21,
      .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .flux1, .hiDreamI1, .hiDreamO1, .zImage,
      .ernieImage, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .qwenImage, .qwenImage2_1, .wan22_5b,
      .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
      return 24_000
    case .longcatVideoAvatar1_5:
      return 16_000
    case .minimaxH3:
      return 32_000
    case .ltx2:
      return 24_000
    case .ltx2_3:
      return 48_000
    }
  }

  public static func stageModelsForModel(_ name: String) -> [String] {
    guard let specification = specificationForModel(name) else { return [] }
    return specification.stageModels ?? []
  }

  public static func MMDiTForModel(_ name: String) -> Specification.MMDiT? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.mmdit
  }

  public static func isUpcastAttentionForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.upcastAttention
  }

  public static func isHighPrecisionAutoencoderForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.highPrecisionAutoencoder ?? false
  }

  public static func modifierForModel(_ name: String) -> SamplerModifier {
    guard let specification = specificationForModel(name) else { return .none }
    return specification.modifier ?? .none
  }

  public static func isConsistencyModelForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.isConsistencyModel ?? false
  }

  public static func guidanceEmbedForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.guidanceEmbed ?? false
  }

  public static func framesPerSecondForModel(_ name: String) -> Double {
    guard let specification = specificationForModel(name) else { return 30 }
    if let framesPerSecond = specification.framesPerSecond {
      return framesPerSecond
    }
    switch specification.version {
    case .hunyuanVideo:
      return 30
    case .longcatVideoAvatar1_5:
      return 25
    case .wan21_1_3b, .wan21_14b, .qwenImage, .qwenImage2_1:
      return 16
    case .wan22_5b, .minimaxH3:
      return 24
    case .ltx2, .ltx2_3:
      return 25
    case .v1, .v2, .auraflow, .flux1, .hiDreamI1, .hiDreamO1, .kandinsky21, .pixart, .sd3,
      .sd3Large,
      .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageB, .wurstchenStageC, .zImage,
      .ernieImage,
      .flux2, .flux2_9b,
      .flux2_4b, .cosmos2_5_2b, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
      return 30
    }
  }

  public static func defaultScaleForModel(_ name: String?) -> UInt16 {
    guard let name = name, let specification = specificationForModel(name) else { return 8 }
    return specification.defaultScale
  }

  public static func hiresFixScaleForModel(_ name: String?) -> UInt16 {
    guard let name = name, let specification = specificationForModel(name) else { return 8 }
    return specification.hiresFixScale ?? specification.defaultScale
  }

  public static func teaCacheCoefficientsForModel(_ name: String) -> (
    Float, Float, Float, Float, Float
  )? {
    guard let specification = specificationForModel(name) else { return nil }
    guard let coefficients = specification.teaCacheCoefficients else {
      switch specification.version {
      case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large, .wan21_1_3b, .wan21_14b, .qwenImage,
        .qwenImage2_1, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b,
        .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .hiDreamO1, .ideogram4, .krea2,
        .longcatVideoAvatar1_5:
        return nil
      case .minimaxH3:
        // FirstBlockCache uses per-modality relative L1 directly, without a fitted polynomial.
        return (0, 0, 0, 1, 0)
      case .flux1:
        return (4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01)
      case .hunyuanVideo:
        return (7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02)
      case .hiDreamI1:
        return (-3.13605009e+04, -7.12425503e+02, 4.91363285e+01, 8.26515490e+00, 1.08053901e-01)
      }
    }
    return (coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4])
  }

  public static func isBF16ForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.isBf16 ?? false
  }

  public static func defaultRefinerForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.defaultRefiner
  }

  public static func mergeFileSHA256(_ sha256: [String: String]) {
    var fileSHA256 = fileSHA256
    for (key, value) in sha256 {
      fileSHA256[key] = value
    }
    self.fileSHA256 = fileSHA256
  }

  public static func fileSHA256ForModelDownloaded(_ name: String) -> String? {
    return fileSHA256[name]
  }

  public static func is8BitModel(_ name: String) -> Bool {
    let filePath = Self.filePathForModelDownloaded(name)
    let fileSize = (try? URL(fileURLWithPath: filePath).resourceValues(forKeys: [.fileSizeKey]))?
      .fileSize
    let externalFileSize =
      (try? URL(fileURLWithPath: TensorData.externalStore(filePath: filePath)).resourceValues(
        forKeys: [.fileSizeKey]))?.fileSize
    if var fileSize = fileSize {
      fileSize += externalFileSize ?? 0
      let version = versionForModel(name)
      switch version {
      case .sdxlBase, .sdxlRefiner:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .ssd1b:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .v1, .v2, .pixart:
        return fileSize < 1_024 * 1_024 * 1_024
      case .kandinsky21:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .svdI2v:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .wurstchenStageC:
        return fileSize < 4 * 1_024 * 1_024 * 1_024
      case .wurstchenStageB:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .sd3:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .sd3Large:
        return fileSize < 7 * 1_024 * 1_024 * 1_024
      case .auraflow:
        return fileSize < 6 * 1_024 * 1_024 * 1_024
      case .flux1:
        return fileSize < 10 * 1_024 * 1_024 * 1_024
      case .hunyuanVideo:
        return fileSize < 11 * 1_024 * 1_024 * 1_024
      case .wan21_1_3b:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .wan22_5b:
        return fileSize < 6 * 1_024 * 1_024 * 1_024
      case .wan21_14b:
        return fileSize < 13 * 1_024 * 1_024 * 1_024 + 512 * 1_024 * 1_024
      case .longcatVideoAvatar1_5:
        return fileSize < 16 * 1_024 * 1_024 * 1_024
      case .qwenImage2_1:
        return fileSize < 12 * 1_024 * 1_024 * 1_024
      case .qwenImage:
        return fileSize < 17 * 1_024 * 1_024 * 1_024 + 512 * 1_024 * 1_024
      case .cosmos2_5_2b:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .hiDreamI1:
        return fileSize < 13 * 1_024 * 1_024 * 1_024 + 512 * 1_024 * 1_024
      case .hiDreamO1:
        return false
      case .zImage:
        return fileSize < 5 * 1_024 * 1_024 * 1_024
      case .ernieImage:
        return fileSize < 9 * 1_024 * 1_024 * 1_024
      case .flux2:
        return fileSize < 26 * 1_024 * 1_024 * 1_024
      case .flux2_9b:
        return fileSize < 8 * 1_024 * 1_024 * 1_024
      case .flux2_4b:
        return fileSize < 4 * 1_024 * 1_024 * 1_024
      case .ltx2, .ltx2_3:
        return fileSize < 20 * 1_024 * 1_024 * 1_024
      case .seedvr2_3b:
        return fileSize < 5 * 1_024 * 1_024 * 1_024
      case .seedvr2_7b:
        return fileSize < 8 * 1_024 * 1_024 * 1_024
      case .ideogram4:
        return fileSize < 9 * 1_024 * 1_024 * 1_024
      case .krea2:
        return fileSize < 9 * 1_024 * 1_024 * 1_024
      case .minimaxH3:
        return fileSize < 28 * 1_024 * 1_024 * 1_024
      }
    }
    return false
  }

  public static func isQuantizedModel(_ name: String) -> Bool {
    let filePath = Self.filePathForModelDownloaded(name)
    let fileSize = (try? URL(fileURLWithPath: filePath).resourceValues(forKeys: [.fileSizeKey]))?
      .fileSize
    let externalFileSize =
      (try? URL(fileURLWithPath: TensorData.externalStore(filePath: filePath)).resourceValues(
        forKeys: [.fileSizeKey]))?.fileSize
    if var fileSize = fileSize {
      fileSize += externalFileSize ?? 0
      let version = versionForModel(name)
      switch version {
      case .sdxlBase, .sdxlRefiner:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .ssd1b:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .v1, .v2, .pixart:
        return fileSize < 1_024 * 1_024 * 1_024
      case .kandinsky21:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .svdI2v:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .wurstchenStageC:
        return fileSize < 4 * 1_024 * 1_024 * 1_024
      case .wurstchenStageB:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .sd3:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .sd3Large:
        return fileSize < 10 * 1_024 * 1_024 * 1_024
      case .auraflow:
        return fileSize < 10 * 1_024 * 1_024 * 1_024
      case .flux1:
        return fileSize < 16 * 1_024 * 1_024 * 1_024
      case .hunyuanVideo:
        return fileSize < 20 * 1_024 * 1_024 * 1_024
      case .wan21_1_3b:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .wan22_5b:
        return fileSize < 6 * 1_024 * 1_024 * 1_024
      case .wan21_14b:
        return fileSize < 15 * 1_024 * 1_024 * 1_024
      case .longcatVideoAvatar1_5:
        return fileSize < 16 * 1_024 * 1_024 * 1_024
      case .qwenImage2_1:
        return fileSize < 12 * 1_024 * 1_024 * 1_024
      case .qwenImage:
        return fileSize < 20 * 1_024 * 1_024 * 1_024
      case .cosmos2_5_2b:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .hiDreamI1:
        return fileSize < 17 * 1_024 * 1_024 * 1_024
      case .hiDreamO1:
        return false
      case .zImage:
        return fileSize < 8 * 1_024 * 1_024 * 1_024
      case .ernieImage:
        return fileSize < 9 * 1_024 * 1_024 * 1_024
      case .flux2:
        return fileSize < 40 * 1_024 * 1_024 * 1_024
      case .flux2_9b:
        return fileSize < 15 * 1_024 * 1_024 * 1_024
      case .flux2_4b:
        return fileSize < 7 * 1_024 * 1_024 * 1_024
      case .ltx2, .ltx2_3:
        return fileSize < 20 * 1_024 * 1_024 * 1_024
      case .seedvr2_3b:
        return fileSize < 5 * 1_024 * 1_024 * 1_024
      case .seedvr2_7b:
        return fileSize < 8 * 1_024 * 1_024 * 1_024
      case .ideogram4:
        return fileSize < 18 * 1_024 * 1_024 * 1_024
      case .krea2:
        return fileSize < 13 * 1_024 * 1_024 * 1_024
      case .minimaxH3:
        return fileSize < 36 * 1_024 * 1_024 * 1_024
      }
    }
    return false
  }
}

extension ModelZoo {
  public static func ensureDownloadsDirectoryExists() {
    let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    let downloadsUrl = urls.first!.appendingPathComponent("Downloads")
    try? FileManager.default.createDirectory(at: downloadsUrl, withIntermediateDirectories: true)
  }

  public static func allDownloadedFiles(
    _ includesSystemDownloadUrl: Bool = true,
    matchingSuffixes: [String] = [
      ".safetensors", ".ckpt", ".ckpt.zip", ".pt", ".pt.zip", ".pth", ".pth.zip", ".bin", ".patch",
    ]
  ) -> [String] {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .documentDirectory, in: .userDomainMask)
    let downloadsUrl = urls.first!.appendingPathComponent("Downloads")
    let systemDownloadUrl = fileManager.urls(for: .downloadsDirectory, in: .userDomainMask).first
    var fileUrls = [URL]()
    if let urls = try? fileManager.contentsOfDirectory(
      at: downloadsUrl, includingPropertiesForKeys: [.fileSizeKey],
      options: [.skipsHiddenFiles, .skipsPackageDescendants, .skipsSubdirectoryDescendants])
    {
      fileUrls.append(contentsOf: urls)
    }
    if includesSystemDownloadUrl,
      let systemDownloadUrl = systemDownloadUrl?.resolvingSymlinksInPath(),
      let urls = try? fileManager.contentsOfDirectory(
        at: systemDownloadUrl, includingPropertiesForKeys: [.fileSizeKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants, .skipsSubdirectoryDescendants])
    {
      // From the system download directory, we only include file with ckpt, pt or safetensors suffix.
      fileUrls.append(
        contentsOf: urls.filter {
          let path = $0.path.lowercased()
          return matchingSuffixes.contains { path.hasSuffix($0) }
        })
    }
    return fileUrls.compactMap {
      guard let values = try? $0.resourceValues(forKeys: [.fileSizeKey]) else { return nil }
      guard let fileSize = values.fileSize, fileSize > 0 else { return nil }
      let file = $0.lastPathComponent
      guard !file.hasSuffix(".part") else { return nil }
      return file
    }
  }

  public static func filePathForDownloadedFile(_ file: String) -> String {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .documentDirectory, in: .userDomainMask)
    let downloadsUrl = urls.first!.appendingPathComponent("Downloads")
    let filePath = downloadsUrl.appendingPathComponent(file).path
    if !fileManager.fileExists(atPath: filePath),
      // Check if the file exists in system download directory.
      let systemDownloadUrl = fileManager.urls(for: .downloadsDirectory, in: .userDomainMask).first
    {
      let systemFilePath = systemDownloadUrl.appendingPathComponent(file).path
      return fileManager.fileExists(atPath: systemFilePath) ? systemFilePath : filePath
    }
    return filePath
  }

  public static func fileBytesForDownloadedFile(_ file: String) -> Int64 {
    let filePath = Self.filePathForDownloadedFile(file)
    let fileSize = (try? URL(fileURLWithPath: filePath).resourceValues(forKeys: [.fileSizeKey]))?
      .fileSize
    return Int64(fileSize ?? 0)
  }

  public static func availableFiles(excluding file: String?) -> Set<String> {
    var files = Set<String>()
    for specification in availableSpecifications {
      guard specification.file != file, ModelZoo.isModelDownloaded(specification.file) else {
        continue
      }
      files.insert(specification.file)
      let textEncoder = specification.textEncoder ?? "clip_vit_l14_f16.ckpt"
      files.insert(textEncoder)
      let autoencoder = specification.autoencoder ?? "vae_ft_mse_840000_f16.ckpt"
      files.insert(autoencoder)
      if let imageEncoder = specification.imageEncoder {
        files.insert(imageEncoder)
      }
      if let clipEncoder = specification.clipEncoder {
        files.insert(clipEncoder)
      }
      specification.additionalClipEncoders?.forEach {
        files.insert($0)
      }
      if let t5Encoder = specification.t5Encoder {
        files.insert(t5Encoder)
      }
      if let diffusionMapping = specification.diffusionMapping {
        files.insert(diffusionMapping)
      }
      if let stageModels = specification.stageModels {
        files.formUnion(stageModels)
      }
      specification.latentsUpscalers?.forEach {
        files.insert($0.file)
      }
    }
    return files
  }

  public static func isResolutionDependentShiftAvailable(
    _ version: ModelVersion, isConsistencyModel: Bool
  ) -> Bool {
    guard
      version == .flux1 || version == .sd3 || version == .sd3Large || version == .hiDreamI1
        || version == .qwenImage || version == .qwenImage2_1 || version == .zImage
        || version == .flux2 || version == .flux2_9b
        || version == .flux2_4b || version == .cosmos2_5_2b || version == .ernieImage
        || version == .ideogram4 || version == .krea2
    else { return false }
    if isConsistencyModel {
      return false
    }
    return true
  }

  public static func shiftFor(_ resolution: (width: UInt16, height: UInt16), version: ModelVersion)
    -> Double
  {
    if version == .qwenImage2_1 {
      return exp(
        ((Double(resolution.height) * Double(resolution.width)) * 16 - 256)
          * (0.9 - 0.5) / (8192 - 256) + 0.5)
    }
    return exp(
      ((Double(resolution.height) * Double(resolution.width)) * 16 - 256)
        * (1.15 - 0.5) / (4096 - 256) + 0.5)
  }

  public static func isCLIPSkipAvailable(_ version: ModelVersion) -> Bool {
    switch version {
    case .pixart, .auraflow, .wan21_14b, .wan21_1_3b, .qwenImage, .qwenImage2_1, .svdI2v, .wan22_5b,
      .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ltx2, .ltx2_3,
      .seedvr2_3b, .seedvr2_7b, .hiDreamO1, .ideogram4, .krea2, .longcatVideoAvatar1_5, .minimaxH3:
      return false
    case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .v1, .v2, .flux1, .hunyuanVideo, .hiDreamI1,
      .ssd1b, .kandinsky21, .wurstchenStageB, .wurstchenStageC:
      return true
    }
  }

  public static func isCompatibleSampler(_ model: String, sampler: SamplerType) -> Bool {
    let objective = objectiveForModel(model)
    switch objective {
    case .edm(_), .v, .epsilon:
      return true
    case .u(_):
      switch sampler {
      case .DPMPP2MAYS, .DPMPPSDEAYS, .dDIMTrailing, .dPMPP2MTrailing, .dPMPPSDETrailing,
        .eulerAAYS, .eulerATrailing, .uniPCAYS, .uniPCTrailing, .tCDTrailing:
        return true
      case .DDIM, .PLMS, .LCM, .TCD, .dPMPP2MKarras, .dPMPPSDEKarras, .dPMPPSDESubstep, .eulerA,
        .eulerASubstep, .uniPC:
        return false
      }
    }
  }

  public static func isCompatibleRefiner(_ version: ModelVersion, refinerVersion: ModelVersion)
    -> Bool
  {
    if version == refinerVersion {
      return true
    }
    if [.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
      && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(refinerVersion)
    {
      return true
    }
    // All uses SD3 VAE.
    if [.sd3, .sd3Large].contains(version) && [.sd3, .sd3Large].contains(refinerVersion) {
      return true
    }
    // All uses Wan VAE.
    if [.wan21_1_3b, .wan21_14b].contains(version)
      && [.wan21_1_3b, .wan21_14b].contains(refinerVersion)
    {
      return true
    }
    /*
    // All uses SDXL VAE.
    if [.sdxlBase, .sdxlRefiner, .ssd1b, .auraflow, .pixart].contains(version)
      && [.sdxlBase, .sdxlRefiner, .ssd1b, .auraflow, .pixart].contains(refinerVersion)
    {
      return true
    }
    // All uses FLUX VAE.
    if [.flux1, .hiDreamI1].contains(version) && [.flux1, .hiDreamI1].contains(refinerVersion) {
      return true
    }
    // All uses SD3 VAE.
    if [.sd3, .sd3Large].contains(version) && [.sd3, .sd3Large].contains(refinerVersion) {
      return true
    }
    // All uses Wan VAE.
    if [.wan21_1_3b, .wan21_14b, .qwenImage, .cosmos2_5_2b].contains(version)
      && [.wan21_1_3b, .wan21_14b, .qwenImage, .cosmos2_5_2b].contains(refinerVersion)
    {
      return true
    }
     */
    return false
  }
}
