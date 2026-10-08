import Foundation

#if canImport(CoreML)
  import CoreML
  import DiffusionCoreML
  import ZIPFoundation
#endif

#if canImport(CoreML)
  extension ImageConverter {
    static let flux1UnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: coreMLUrl.appendingPathComponent("flux_1_tae").path)
        else {
          return coreMLUrl.appendingPathComponent("flux_1_tae")
        }
        try fileManager.unzipItem(
          at: Bundle.main.url(
            forResource: "flux_1_tae", withExtension: "zip")!,
          to: coreMLUrl)
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/768.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/768.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/1024.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/1024.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/1280.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/1280.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/1536.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/1536.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/1792.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_1_tae/1792.mlmodelc/weights/weight.bin"))
        }
        try fileManager.moveItem(
          at: coreMLUrl.appendingPathComponent("flux_1_tae/weight.bin"),
          to: coreMLUrl.appendingPathComponent("flux_1_tae/2048.mlmodelc/weights/weight.bin"))
        return coreMLUrl.appendingPathComponent("flux_1_tae")
      } catch {
        try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent("flux_1_tae"))
        return nil
      }
    }()

    static let sdUnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: coreMLUrl.appendingPathComponent("sd_tae").path)
        else {
          return coreMLUrl.appendingPathComponent("sd_tae")
        }
        try fileManager.unzipItem(
          at: Bundle.main.url(
            forResource: "sd_tae", withExtension: "zip")!,
          to: coreMLUrl)
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/768.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/768.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/1024.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/1024.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/1280.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/1280.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/1536.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/1536.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/1792.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd_tae/1792.mlmodelc/weights/weight.bin"))
        }
        try fileManager.moveItem(
          at: coreMLUrl.appendingPathComponent("sd_tae/weight.bin"),
          to: coreMLUrl.appendingPathComponent("sd_tae/2048.mlmodelc/weights/weight.bin"))
        return coreMLUrl.appendingPathComponent("sd_tae")
      } catch {
        try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent("sd_tae"))
        return nil
      }
    }()

    static let sd3UnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: coreMLUrl.appendingPathComponent("sd3_tae").path)
        else {
          return coreMLUrl.appendingPathComponent("sd3_tae")
        }
        try fileManager.unzipItem(
          at: Bundle.main.url(
            forResource: "sd3_tae", withExtension: "zip")!,
          to: coreMLUrl)
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/768.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/768.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/1024.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/1024.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/1280.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/1280.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/1536.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/1536.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/1792.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sd3_tae/1792.mlmodelc/weights/weight.bin"))
        }
        try fileManager.moveItem(
          at: coreMLUrl.appendingPathComponent("sd3_tae/weight.bin"),
          to: coreMLUrl.appendingPathComponent("sd3_tae/2048.mlmodelc/weights/weight.bin"))
        return coreMLUrl.appendingPathComponent("sd3_tae")
      } catch {
        try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent("sd3_tae"))
        return nil
      }
    }()

    static let sdxlUnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: coreMLUrl.appendingPathComponent("sdxl_tae").path)
        else {
          return coreMLUrl.appendingPathComponent("sdxl_tae")
        }
        try fileManager.unzipItem(
          at: Bundle.main.url(
            forResource: "sdxl_tae", withExtension: "zip")!,
          to: coreMLUrl)
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/768.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/768.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/1024.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/1024.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/1280.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/1280.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/1536.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/1536.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/1792.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("sdxl_tae/1792.mlmodelc/weights/weight.bin"))
        }
        try fileManager.moveItem(
          at: coreMLUrl.appendingPathComponent("sdxl_tae/weight.bin"),
          to: coreMLUrl.appendingPathComponent("sdxl_tae/2048.mlmodelc/weights/weight.bin"))
        return coreMLUrl.appendingPathComponent("sdxl_tae")
      } catch {
        try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent("sdxl_tae"))
        return nil
      }
    }()

    static let qwenImageUnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard
          !fileManager.fileExists(atPath: coreMLUrl.appendingPathComponent("qwenimage_tae").path)
        else {
          return coreMLUrl.appendingPathComponent("qwenimage_tae")
        }
        try fileManager.unzipItem(
          at: Bundle.main.url(
            forResource: "qwenimage_tae", withExtension: "zip")!,
          to: coreMLUrl)
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/768.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/768.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/1024.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/1024.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/1280.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/1280.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/1536.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/1536.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/1792.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("qwenimage_tae/1792.mlmodelc/weights/weight.bin"))
        }
        try fileManager.moveItem(
          at: coreMLUrl.appendingPathComponent("qwenimage_tae/weight.bin"),
          to: coreMLUrl.appendingPathComponent("qwenimage_tae/2048.mlmodelc/weights/weight.bin"))
        return coreMLUrl.appendingPathComponent("qwenimage_tae")
      } catch {
        try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent("qwenimage_tae"))
        return nil
      }
    }()

    static let qwenImage2_1UnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      let modelUrl = coreMLUrl.appendingPathComponent("qwen_image_2_1_tae")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: modelUrl.path) else { return modelUrl }
        guard
          let archiveUrl = Bundle.main.url(forResource: "qwen_image_2.1_tae", withExtension: "zip")
        else { return nil }
        try fileManager.unzipItem(at: archiveUrl, to: coreMLUrl)
        for size in [768, 1024, 1280, 1536, 1792] {
          do {
            try fileManager.linkItem(
              at: modelUrl.appendingPathComponent("weight.bin"),
              to: modelUrl.appendingPathComponent("\(size).mlmodelc/weights/weight.bin"))
          } catch {
            try fileManager.copyItem(
              at: modelUrl.appendingPathComponent("weight.bin"),
              to: modelUrl.appendingPathComponent("\(size).mlmodelc/weights/weight.bin"))
          }
        }
        try fileManager.moveItem(
          at: modelUrl.appendingPathComponent("weight.bin"),
          to: modelUrl.appendingPathComponent("2048.mlmodelc/weights/weight.bin"))
        return modelUrl
      } catch {
        try? fileManager.removeItem(at: modelUrl)
        return nil
      }
    }()

    static let wan21UnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: coreMLUrl.appendingPathComponent("wan_2.1_tae").path)
        else {
          return coreMLUrl.appendingPathComponent("wan_2.1_tae")
        }
        guard
          let qwenImageTAEArchiveURL = Bundle.main.url(
            forResource: "qwenimage_tae", withExtension: "zip"),
          let wan21TAEArchiveURL = Bundle.main.url(
            forResource: "wan_2.1_tae", withExtension: "zip")
        else {
          return nil
        }
        guard let qwenImageTAEArchive = Archive(url: qwenImageTAEArchiveURL, accessMode: .read),
          let qwenImageWeight = qwenImageTAEArchive["qwenimage_tae/weight.bin"]
        else {
          return nil
        }
        try fileManager.unzipItem(at: wan21TAEArchiveURL, to: coreMLUrl)
        var qwenImageWeightData = Data()
        let _ = try qwenImageTAEArchive.extract(qwenImageWeight) { qwenImageWeightData.append($0) }
        let weightDiffUrl = coreMLUrl.appendingPathComponent("wan_2.1_tae/weight.bin.insdiff")
        let wan21PatchData = try Data(contentsOf: weightDiffUrl)
        let weightData = try InsdiffPatcher.apply(
          baseData: qwenImageWeightData, patchData: wan21PatchData)
        let weightUrl = coreMLUrl.appendingPathComponent("wan_2.1_tae/weight.bin")
        try weightData.write(to: weightUrl, options: .atomic)
        try fileManager.removeItem(at: weightDiffUrl)

        do {
          try fileManager.linkItem(
            at: weightUrl,
            to: coreMLUrl.appendingPathComponent("wan_2.1_tae/512.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: weightUrl,
            to: coreMLUrl.appendingPathComponent("wan_2.1_tae/512.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: weightUrl,
            to: coreMLUrl.appendingPathComponent("wan_2.1_tae/768.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: weightUrl,
            to: coreMLUrl.appendingPathComponent("wan_2.1_tae/768.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: weightUrl,
            to: coreMLUrl.appendingPathComponent("wan_2.1_tae/1024.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: weightUrl,
            to: coreMLUrl.appendingPathComponent("wan_2.1_tae/1024.mlmodelc/weights/weight.bin"))
        }
        try fileManager.moveItem(
          at: weightUrl,
          to: coreMLUrl.appendingPathComponent("wan_2.1_tae/1280.mlmodelc/weights/weight.bin"))
        return coreMLUrl.appendingPathComponent("wan_2.1_tae")
      } catch {
        try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent("wan_2.1_tae"))
        return nil
      }
    }()

    static let flux2UnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: coreMLUrl.appendingPathComponent("flux_2_tae").path)
        else {
          return coreMLUrl.appendingPathComponent("flux_2_tae")
        }
        try fileManager.unzipItem(
          at: Bundle.main.url(
            forResource: "flux_2_tae", withExtension: "zip")!,
          to: coreMLUrl)
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/768.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/768.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/1024.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/1024.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/1280.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/1280.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/1536.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/1536.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/1792.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("flux_2_tae/1792.mlmodelc/weights/weight.bin"))
        }
        try fileManager.moveItem(
          at: coreMLUrl.appendingPathComponent("flux_2_tae/weight.bin"),
          to: coreMLUrl.appendingPathComponent("flux_2_tae/2048.mlmodelc/weights/weight.bin"))
        return coreMLUrl.appendingPathComponent("flux_2_tae")
      } catch {
        try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent("flux_2_tae"))
        return nil
      }
    }()

    static let MiniMaxH3UnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      let modelUrl = coreMLUrl.appendingPathComponent("minimax_h3_tae")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: modelUrl.path) else { return modelUrl }
        guard let archiveUrl = Bundle.main.url(forResource: "minimax_h3_tae", withExtension: "zip")
        else { return nil }
        try fileManager.unzipItem(at: archiveUrl, to: coreMLUrl)
        for size in [768, 1024, 1280, 1536, 1792] {
          do {
            try fileManager.linkItem(
              at: modelUrl.appendingPathComponent("weight.bin"),
              to: modelUrl.appendingPathComponent("\(size).mlmodelc/weights/weight.bin"))
          } catch {
            try fileManager.copyItem(
              at: modelUrl.appendingPathComponent("weight.bin"),
              to: modelUrl.appendingPathComponent("\(size).mlmodelc/weights/weight.bin"))
          }
        }
        try fileManager.moveItem(
          at: modelUrl.appendingPathComponent("weight.bin"),
          to: modelUrl.appendingPathComponent("2048.mlmodelc/weights/weight.bin"))
        return modelUrl
      } catch {
        try? fileManager.removeItem(at: modelUrl)
        return nil
      }
    }()

    static let LTX2UnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: coreMLUrl.appendingPathComponent("ltx_2_tae").path)
        else {
          return coreMLUrl.appendingPathComponent("ltx_2_tae")
        }
        try fileManager.unzipItem(
          at: Bundle.main.url(
            forResource: "ltx_2_tae", withExtension: "zip")!,
          to: coreMLUrl)
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/768.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/768.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/1024.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/1024.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/1280.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/1280.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/1536.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/1536.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/1792.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2_tae/1792.mlmodelc/weights/weight.bin"))
        }
        try fileManager.moveItem(
          at: coreMLUrl.appendingPathComponent("ltx_2_tae/weight.bin"),
          to: coreMLUrl.appendingPathComponent("ltx_2_tae/2048.mlmodelc/weights/weight.bin"))
        return coreMLUrl.appendingPathComponent("ltx_2_tae")
      } catch {
        try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent("ltx_2_tae"))
        return nil
      }
    }()

    static let LTX2_3UnzipItem: URL? = {
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      do {
        try fileManager.createDirectory(at: coreMLUrl, withIntermediateDirectories: true)
        guard !fileManager.fileExists(atPath: coreMLUrl.appendingPathComponent("ltx_2.3_tae").path)
        else {
          return coreMLUrl.appendingPathComponent("ltx_2.3_tae")
        }
        try fileManager.unzipItem(
          at: Bundle.main.url(
            forResource: "ltx_2.3_tae", withExtension: "zip")!,
          to: coreMLUrl)
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/768.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/768.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/1024.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/1024.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/1280.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/1280.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/1536.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/1536.mlmodelc/weights/weight.bin"))
        }
        do {
          try fileManager.linkItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/1792.mlmodelc/weights/weight.bin"))
        } catch {
          try fileManager.copyItem(
            at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
            to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/1792.mlmodelc/weights/weight.bin"))
        }
        try fileManager.moveItem(
          at: coreMLUrl.appendingPathComponent("ltx_2.3_tae/weight.bin"),
          to: coreMLUrl.appendingPathComponent("ltx_2.3_tae/2048.mlmodelc/weights/weight.bin"))
        return coreMLUrl.appendingPathComponent("ltx_2.3_tae")
      } catch {
        try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent("ltx_2.3_tae"))
        return nil
      }
    }()
    static let flux1TinyDecoderFor768: ManagedMLModel? = {
      guard let flux1UnzipItem = flux1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let flux1TinyDecoderFor1024: ManagedMLModel? = {
      guard let flux1UnzipItem = flux1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let flux1TinyDecoderFor1280: ManagedMLModel? = {
      guard let flux1UnzipItem = flux1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let flux1TinyDecoderFor1536: ManagedMLModel? = {
      guard let flux1UnzipItem = flux1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let flux1TinyDecoderFor1792: ManagedMLModel? = {
      guard let flux1UnzipItem = flux1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let flux1TinyDecoderFor2048: ManagedMLModel? = {
      guard let flux1UnzipItem = flux1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
    static let sdTinyDecoderFor768: ManagedMLModel? = {
      guard let sdUnzipItem = sdUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdUnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let sdTinyDecoderFor1024: ManagedMLModel? = {
      guard let sdUnzipItem = sdUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdUnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let sdTinyDecoderFor1280: ManagedMLModel? = {
      guard let sdUnzipItem = sdUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdUnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let sdTinyDecoderFor1536: ManagedMLModel? = {
      guard let sdUnzipItem = sdUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdUnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let sdTinyDecoderFor1792: ManagedMLModel? = {
      guard let sdUnzipItem = sdUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdUnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let sdTinyDecoderFor2048: ManagedMLModel? = {
      guard let sdUnzipItem = sdUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdUnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
    static let sd3TinyDecoderFor768: ManagedMLModel? = {
      guard let sd3UnzipItem = sd3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sd3UnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let sd3TinyDecoderFor1024: ManagedMLModel? = {
      guard let sd3UnzipItem = sd3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sd3UnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let sd3TinyDecoderFor1280: ManagedMLModel? = {
      guard let sd3UnzipItem = sd3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sd3UnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let sd3TinyDecoderFor1536: ManagedMLModel? = {
      guard let sd3UnzipItem = sd3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sd3UnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let sd3TinyDecoderFor1792: ManagedMLModel? = {
      guard let sd3UnzipItem = sd3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sd3UnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let sd3TinyDecoderFor2048: ManagedMLModel? = {
      guard let sd3UnzipItem = sd3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sd3UnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
    static let sdxlTinyDecoderFor768: ManagedMLModel? = {
      guard let sdxlUnzipItem = sdxlUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdxlUnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let sdxlTinyDecoderFor1024: ManagedMLModel? = {
      guard let sdxlUnzipItem = sdxlUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdxlUnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let sdxlTinyDecoderFor1280: ManagedMLModel? = {
      guard let sdxlUnzipItem = sdxlUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdxlUnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let sdxlTinyDecoderFor1536: ManagedMLModel? = {
      guard let sdxlUnzipItem = sdxlUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdxlUnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let sdxlTinyDecoderFor1792: ManagedMLModel? = {
      guard let sdxlUnzipItem = sdxlUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdxlUnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let sdxlTinyDecoderFor2048: ManagedMLModel? = {
      guard let sdxlUnzipItem = sdxlUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: sdxlUnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImageTinyDecoderFor768: ManagedMLModel? = {
      guard let flux1UnzipItem = qwenImageUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImageTinyDecoderFor1024: ManagedMLModel? = {
      guard let flux1UnzipItem = qwenImageUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImageTinyDecoderFor1280: ManagedMLModel? = {
      guard let flux1UnzipItem = qwenImageUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImageTinyDecoderFor1536: ManagedMLModel? = {
      guard let flux1UnzipItem = qwenImageUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImageTinyDecoderFor1792: ManagedMLModel? = {
      guard let flux1UnzipItem = qwenImageUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImageTinyDecoderFor2048: ManagedMLModel? = {
      guard let flux1UnzipItem = qwenImageUnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux1UnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImage2_1TinyDecoderFor768: ManagedMLModel? = {
      guard let qwenImage2_1UnzipItem = qwenImage2_1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: qwenImage2_1UnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImage2_1TinyDecoderFor1024: ManagedMLModel? = {
      guard let qwenImage2_1UnzipItem = qwenImage2_1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: qwenImage2_1UnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImage2_1TinyDecoderFor1280: ManagedMLModel? = {
      guard let qwenImage2_1UnzipItem = qwenImage2_1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: qwenImage2_1UnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImage2_1TinyDecoderFor1536: ManagedMLModel? = {
      guard let qwenImage2_1UnzipItem = qwenImage2_1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: qwenImage2_1UnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImage2_1TinyDecoderFor1792: ManagedMLModel? = {
      guard let qwenImage2_1UnzipItem = qwenImage2_1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: qwenImage2_1UnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let qwenImage2_1TinyDecoderFor2048: ManagedMLModel? = {
      guard let qwenImage2_1UnzipItem = qwenImage2_1UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: qwenImage2_1UnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
    static let wan21TinyDecoderFor512: ManagedMLModel? = {
      guard let wan21UnzipItem = wan21UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: wan21UnzipItem.appendingPathComponent("512.mlmodelc"),
        configuration: configuration)
    }()
    static let wan21TinyDecoderFor768: ManagedMLModel? = {
      guard let wan21UnzipItem = wan21UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: wan21UnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let wan21TinyDecoderFor1024: ManagedMLModel? = {
      guard let wan21UnzipItem = wan21UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: wan21UnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let wan21TinyDecoderFor1280: ManagedMLModel? = {
      guard let wan21UnzipItem = wan21UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: wan21UnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let flux2TinyDecoderFor768: ManagedMLModel? = {
      guard let flux2UnzipItem = flux2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux2UnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let flux2TinyDecoderFor1024: ManagedMLModel? = {
      guard let flux2UnzipItem = flux2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux2UnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let flux2TinyDecoderFor1280: ManagedMLModel? = {
      guard let flux2UnzipItem = flux2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux2UnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let flux2TinyDecoderFor1536: ManagedMLModel? = {
      guard let flux2UnzipItem = flux2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux2UnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let flux2TinyDecoderFor1792: ManagedMLModel? = {
      guard let flux2UnzipItem = flux2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux2UnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let flux2TinyDecoderFor2048: ManagedMLModel? = {
      guard let flux2UnzipItem = flux2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: flux2UnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
    static let MiniMaxH3TinyDecoderFor768: ManagedMLModel? = {
      guard let MiniMaxH3UnzipItem = MiniMaxH3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: MiniMaxH3UnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let MiniMaxH3TinyDecoderFor1024: ManagedMLModel? = {
      guard let MiniMaxH3UnzipItem = MiniMaxH3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: MiniMaxH3UnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let MiniMaxH3TinyDecoderFor1280: ManagedMLModel? = {
      guard let MiniMaxH3UnzipItem = MiniMaxH3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: MiniMaxH3UnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let MiniMaxH3TinyDecoderFor1536: ManagedMLModel? = {
      guard let MiniMaxH3UnzipItem = MiniMaxH3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: MiniMaxH3UnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let MiniMaxH3TinyDecoderFor1792: ManagedMLModel? = {
      guard let MiniMaxH3UnzipItem = MiniMaxH3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: MiniMaxH3UnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let MiniMaxH3TinyDecoderFor2048: ManagedMLModel? = {
      guard let MiniMaxH3UnzipItem = MiniMaxH3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: MiniMaxH3UnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2TinyDecoderFor768: ManagedMLModel? = {
      guard let LTX2UnzipItem = LTX2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2UnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2TinyDecoderFor1024: ManagedMLModel? = {
      guard let LTX2UnzipItem = LTX2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2UnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2TinyDecoderFor1280: ManagedMLModel? = {
      guard let LTX2UnzipItem = LTX2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2UnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2TinyDecoderFor1536: ManagedMLModel? = {
      guard let LTX2UnzipItem = LTX2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2UnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2TinyDecoderFor1792: ManagedMLModel? = {
      guard let LTX2UnzipItem = LTX2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2UnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2TinyDecoderFor2048: ManagedMLModel? = {
      guard let LTX2UnzipItem = LTX2UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2UnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2_3TinyDecoderFor768: ManagedMLModel? = {
      guard let LTX2_3UnzipItem = LTX2_3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2_3UnzipItem.appendingPathComponent("768.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2_3TinyDecoderFor1024: ManagedMLModel? = {
      guard let LTX2_3UnzipItem = LTX2_3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2_3UnzipItem.appendingPathComponent("1024.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2_3TinyDecoderFor1280: ManagedMLModel? = {
      guard let LTX2_3UnzipItem = LTX2_3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2_3UnzipItem.appendingPathComponent("1280.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2_3TinyDecoderFor1536: ManagedMLModel? = {
      guard let LTX2_3UnzipItem = LTX2_3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2_3UnzipItem.appendingPathComponent("1536.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2_3TinyDecoderFor1792: ManagedMLModel? = {
      guard let LTX2_3UnzipItem = LTX2_3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2_3UnzipItem.appendingPathComponent("1792.mlmodelc"),
        configuration: configuration)
    }()
    static let LTX2_3TinyDecoderFor2048: ManagedMLModel? = {
      guard let LTX2_3UnzipItem = LTX2_3UnzipItem else { return nil }
      var configuration = MLModelConfiguration()
      if #available(iOS 16.0, *) {
        configuration.computeUnits = .cpuAndNeuralEngine
      } else {
        configuration.computeUnits = .all
      }
      return ManagedMLModel(
        contentsOf: LTX2_3UnzipItem.appendingPathComponent("2048.mlmodelc"),
        configuration: configuration)
    }()
  }
#endif
