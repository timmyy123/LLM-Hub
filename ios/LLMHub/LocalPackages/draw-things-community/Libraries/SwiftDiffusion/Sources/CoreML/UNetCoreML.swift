import Algorithms
import Atomics
import CoreML
import Diffusion
import DiffusionCoreMLModelManager
import Foundation
import NNC
import NNCCoreMLConversion
import WeightsCache
import ZIPFoundation

public struct UNetFromCoreML<FloatType: TensorNumeric & BinaryFloatingPoint>: UNetProtocol {
  var unetChunk1: ManagedMLModel? = nil
  var unetChunk2: ManagedMLModel? = nil
  public private(set) var version: ModelVersion = .v1
  public init() {}
  public var isLoaded: Bool { unetChunk1 != nil && unetChunk2 != nil }
  public let didRunLoRASeparately: Bool = false
  public func unloadResources() {
    unetChunk1?.unloadResources()
    unetChunk2?.unloadResources()
  }
}

struct TensorNameAndBlobOffset {
  var offset: Int
  var isFirstChunk: Bool
  var isLayerNormBias: Bool
}

extension UNetFromCoreML {
  public var model: AnyModel? { nil }
  public var modelAndWeightMapper: (AnyModel, ModelWeightMapper)? { nil }
  public mutating func unloadModel() {}
  public mutating func compileModel(
    filePath: String, externalOnDemand: Bool, deviceProperties: DeviceProperties,
    version: ModelVersion,
    modifier: SamplerModifier,
    qkNorm: Bool, dualAttentionLayers: [Int], upcastAttention: Bool,
    usesFlashAttention: UseFlashAttention,
    usesSolAttention: Bool,
    solAttentionStart: Int, solAttentionTau: Float,
    injectControlsAndAdapters: InjectControlsAndAdapters<FloatType>, lora: [LoRAConfiguration],
    isQuantizedModel: Bool, canRunLoRASeparately: Bool, inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>?, _ c: [DynamicGraph.AnyTensor],
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: InjectedControlsAndAdapters<FloatType>, referenceImageCount: Int,
    referenceAudioCount: Int,
    tiledDiffusion: TiledConfiguration, teaCache: TeaCacheConfiguration,
    causalInference: (Int, pad: Int), isBF16: Bool, activationQkScaling: [Int: Int],
    activationProjScaling: [Int: Int], activationFfnProjUpScaling: [Int: Int],
    activationFfnScaling: [Int: Int],
    weightsCache: WeightsCache
  ) -> Bool {
    #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
      // We cannot handle upcast attention, yet.
      self.version = version
      let c = c[0]
      guard version != .kandinsky21, upcastAttention == false,
        injectControlsAndAdapters.injectControls == false,
        injectControlsAndAdapters.injectT2IAdapters == false,
        injectControlsAndAdapters.injectIPAdapterLengths.isEmpty,
        CoreMLModelManager.isCoreMLSupported.load(ordering: .acquiring),
        tokenLengthUncond == tokenLengthCond, !tiledDiffusion.isEnabled
      else { return false }
      let isLoRASupported = CoreMLModelManager.isLoRASupported.load(ordering: .acquiring)
      let tokenLength = c.shape[1]
      let startHeight = xT.shape[1]
      let startWidth = xT.shape[2]
      let channels = xT.shape[3]
      guard
        tokenLength == 77 && startWidth == 64 && startHeight == 64
          && (lora.count == 0 || isLoRASupported)
          && (channels == 4 || channels == 9 || channels == 5 || channels == 8)
      else {
        return false
      }
      guard unetChunk1 == nil || unetChunk2 == nil else { return true }
      let components: [String] =
        ([(filePath as NSString).lastPathComponent]
          + lora.flatMap {
            [
              ($0.file as NSString).lastPathComponent,
              ($0.weight * 100).formatted(.number.precision(.fractionLength(0))),
            ]
          })
      let file = String(components.joined(by: "_"))
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let coreMLUrl = urls.first!.appendingPathComponent("coreml")
      let fileSpecificMLUrl = coreMLUrl.appendingPathComponent(file).appendingPathComponent("8x8")
      let firstChunkPath = fileSpecificMLUrl.appendingPathComponent("UnetChunk1.mlmodelc").path
      let secondChunkPath = fileSpecificMLUrl.appendingPathComponent("UnetChunk2.mlmodelc").path
      let isModelConverted =
        CoreMLModelManager.isModelConverted(file) && fileManager.fileExists(atPath: firstChunkPath)
        && fileManager.fileExists(atPath: secondChunkPath)
      var milImmediate = ""
      let lora = lora.filter { $0.version == version }
      if !isModelConverted {
        // Check if we have too much models converted.
        let directories = (try? fileManager.contentsOfDirectory(atPath: coreMLUrl.path)) ?? []
        let maxNumberOfConvertedModels = CoreMLModelManager.maxNumberOfConvertedModels.load(
          ordering: .acquiring)
        if directories.count > maxNumberOfConvertedModels {
          var unusedYet = [String]()
          var used = [String]()
          for directory in directories {
            let model = (directory as NSString).lastPathComponent
            guard file != model else { continue }
            if !CoreMLModelManager.isModelConverted(model) {
              unusedYet.append(directory)
            } else {
              used.append(directory)
            }
          }
          if used.count + 1 > maxNumberOfConvertedModels {  // If we still have more, need to select them out and add to unused list.
            let toBeRemoved = used.randomSample(count: used.count + 1 - maxNumberOfConvertedModels)
            unusedYet.append(contentsOf: toBeRemoved)
            CoreMLModelManager.removeModelsConverted(
              toBeRemoved.map { ($0 as NSString).lastPathComponent })
          }
          if unusedYet.count > directories.count + 1 - maxNumberOfConvertedModels {
            unusedYet = unusedYet.randomSample(
              count: directories.count + 1 - maxNumberOfConvertedModels)
          }
          for directory in unusedYet {
            try? fileManager.removeItem(at: coreMLUrl.appendingPathComponent(directory))
          }
        }
        try? fileManager.createDirectory(at: fileSpecificMLUrl, withIntermediateDirectories: true)
        let blobOffsets: [String: TensorNameAndBlobOffset]
        let milProg2Prefix: Int
        let milProg2Suffix: Int
        if version == .v2 {
          if channels == 9 {
            try? fileManager.unzipItem(
              at: Bundle.main.url(
                forResource: "coreml_sd_v2_inpainting_f16", withExtension: "zip")!,
              to: fileSpecificMLUrl)
            blobOffsets = blobOffsetsSDv2xInpainting
          } else if channels == 5 {
            try? fileManager.unzipItem(
              at: Bundle.main.url(forResource: "coreml_sd_v2_depth_f16", withExtension: "zip")!,
              to: fileSpecificMLUrl)
            blobOffsets = blobOffsetsSDv2xDepth
          } else {
            try? fileManager.unzipItem(
              at: Bundle.main.url(forResource: "coreml_sd_v2_f16", withExtension: "zip")!,
              to: fileSpecificMLUrl)
            blobOffsets = blobOffsetsSDv2x
          }
          milProg2Prefix = 1_017_390
          milProg2Suffix = 584
        } else {
          if channels == 9 {
            try? fileManager.unzipItem(
              at: Bundle.main.url(
                forResource: "coreml_sd_v1.5_inpainting_f16", withExtension: "zip")!,
              to: fileSpecificMLUrl)
            blobOffsets = blobOffsetsSDv1xInpainting
          } else if channels == 8 {
            try? fileManager.unzipItem(
              at: Bundle.main.url(
                forResource: "coreml_sd_v1.5_edit_f16", withExtension: "zip")!,
              to: fileSpecificMLUrl)
            blobOffsets = blobOffsetsSDv1xEdit
          } else {
            try? fileManager.unzipItem(
              at: Bundle.main.url(forResource: "coreml_sd_v1.5_f16", withExtension: "zip")!,
              to: fileSpecificMLUrl)
            blobOffsets = blobOffsetsSDv1x
          }
          milProg2Prefix = 828_980
          milProg2Suffix = 574
        }
        let firstWeight = fopen(firstChunkPath + "/weights/weight.bin", "r+b")
        let secondWeight = fopen(secondChunkPath + "/weights/weight.bin", "r+b")
        let graph = xT.graph
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            graph.openStore(
              filePath, flags: .readOnly,
              externalStore: TensorData.externalStore(filePath: filePath)
            ) { store in
              for key in store.keys {
                let tensor: AnyTensor?
                guard let tensorShape = store.read(like: key) else { continue }
                switch loader.mergeLoRA(
                  graph, name: key, store: store, dataType: Float16.dataType,
                  shape: tensorShape.shape, of: Float16.self)
                {
                case .continue(let name, _, _):
                  tensor = store.read(name, codec: [.q6p, .q8p, .ezm7, .externalData])
                case .final(let x):
                  tensor = x
                case .fail:
                  tensor = nil
                }
                guard let tensor = tensor else { continue }
                var f16tensor = Tensor<Float16>(from: tensor)
                if let blobOffset = blobOffsets[key] {
                  if blobOffset.isLayerNormBias {
                    let bias = graph.constant(f16tensor.toGPU())
                    let weight = graph.constant(
                      Tensor<Float16>(
                        from: store.read(
                          key.dropLast(2) + "0]", codec: [.q6p, .q8p, .ezm7, .externalData])!
                      ).toGPU())
                    f16tensor = Functional.div(left: bias, right: weight).rawValue.toCPU()
                  } else {
                    f16tensor = f16tensor.toCPU()
                  }
                  f16tensor.withUnsafeBytes {
                    guard let u8 = $0.baseAddress else { return }
                    if blobOffset.isFirstChunk {
                      fseek(firstWeight, blobOffset.offset, SEEK_SET)
                      fwrite(u8, 1, $0.count, firstWeight)
                    } else {
                      fseek(secondWeight, blobOffset.offset, SEEK_SET)
                      fwrite(u8, 1, $0.count, secondWeight)
                    }
                  }
                } else if key == "__unet__[t-406-1]" {  // These are immediate values, need to modify the program.
                  f16tensor = f16tensor.toCPU()
                  milImmediate = String(
                    format: "%a, %a, %a, %a", Double(f16tensor[0]), Double(f16tensor[1]),
                    Double(f16tensor[2]), Double(f16tensor[3]))
                }
              }
            }
          }
        } else {
          graph.openStore(
            filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
          ) {
            for key in $0.keys {
              guard let tensor = $0.read(key, codec: [.q6p, .q8p, .ezm7, .externalData]) else {
                continue
              }
              var f16tensor = Tensor<Float16>(from: tensor)
              if let blobOffset = blobOffsets[key] {
                if blobOffset.isLayerNormBias {
                  let bias = graph.constant(f16tensor.toGPU())
                  let weight = graph.constant(
                    Tensor<Float16>(
                      from: $0.read(
                        key.dropLast(2) + "0]", codec: [.q6p, .q8p, .ezm7, .externalData])!
                    ).toGPU())
                  f16tensor = Functional.div(left: bias, right: weight).rawValue.toCPU()
                } else {
                  f16tensor = f16tensor.toCPU()
                }
                f16tensor.withUnsafeBytes {
                  guard let u8 = $0.baseAddress else { return }
                  if blobOffset.isFirstChunk {
                    fseek(firstWeight, blobOffset.offset, SEEK_SET)
                    fwrite(u8, 1, $0.count, firstWeight)
                  } else {
                    fseek(secondWeight, blobOffset.offset, SEEK_SET)
                    fwrite(u8, 1, $0.count, secondWeight)
                  }
                }
              } else if key == "__unet__[t-406-1]" {  // These are immediate values, need to modify the program.
                f16tensor = f16tensor.toCPU()
                milImmediate = String(
                  format: "%a, %a, %a, %a", Double(f16tensor[0]), Double(f16tensor[1]),
                  Double(f16tensor[2]), Double(f16tensor[3]))
              }
            }
          }
        }
        fclose(firstWeight)
        fclose(secondWeight)
        do {
          let oldMilProg2 = try String(
            contentsOf: URL(fileURLWithPath: secondChunkPath + "/model.mil"), encoding: .utf8)
          let newMilProg2 =
            String(oldMilProg2.prefix(milProg2Prefix)) + milImmediate
            + String(oldMilProg2.suffix(milProg2Suffix))
          if newMilProg2 != oldMilProg2 {
            try newMilProg2.write(
              to: URL(fileURLWithPath: secondChunkPath + "/model.mil"), atomically: false,
              encoding: .utf8
            )
          }
        } catch {
          return false
        }
      }
      do {
        let configuration = MLModelConfiguration()
        if #available(iOS 16.0, macOS 13.0, *) {
          let computeUnits = CoreMLModelManager.computeUnits.load(ordering: .acquiring)
          switch computeUnits {
          case 1:
            configuration.computeUnits = .cpuAndGPU
          case 2:
            configuration.computeUnits = .all
          default:  // Including 0
            configuration.computeUnits = .cpuAndNeuralEngine
          }
        }
        let unetChunk1 = ManagedMLModel(
          contentsOf: URL(fileURLWithPath: firstChunkPath), configuration: configuration)
        try unetChunk1.loadResources()
        unetChunk1.unloadResources()
        self.unetChunk1 = unetChunk1
        let unetChunk2 = ManagedMLModel(
          contentsOf: URL(fileURLWithPath: secondChunkPath), configuration: configuration)
        try unetChunk2.loadResources()
        unetChunk2.unloadResources()
        self.unetChunk2 = unetChunk2
        CoreMLModelManager.setModelConverted(file)
        return true
      } catch {
        unetChunk1 = nil
        unetChunk2 = nil
        return false
      }
    #else
      return false
    #endif
  }

  public func callAsFunction(
    timestep: (now: Float, next: Float), audioShiftRatio: Float,
    inputs xT: DynamicGraph.Tensor<FloatType>, _: DynamicGraph.Tensor<FloatType>?,
    _ c: [DynamicGraph.AnyTensor], extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ restInputs: [DynamicGraph.AnyTensor],
      _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [NNC.DynamicGraph.Tensor<FloatType>]
    ),
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>], referenceImageCount: Int,
    referenceAudioCount: Int, step: Int,
    tokenLengthUncond: Int, tokenLengthCond: Int,
    isCfgEnabled: Bool, tiledDiffusion: TiledConfiguration,
    controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    return autoreleasepool {
      let c = c[0]
      let graph = xT.graph
      let unetChunk1 = unetChunk1!
      let unetChunk2 = unetChunk2!
      let reduceMemory: Bool
      switch version {
      case .v1:
        reduceMemory = CoreMLModelManager.reduceMemoryFor1x.load(ordering: .acquiring)
      case .v2:
        reduceMemory = CoreMLModelManager.reduceMemoryFor2x.load(ordering: .acquiring)
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .sdxlBase, .sdxlRefiner,
        .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
        .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage, .ernieImage, .flux2,
        .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4,
        .krea2, .longcatVideoAvatar1_5, .minimaxH3:
        fatalError()
      }
      let channels = xT.shape[3]
      let insideBatch = channels == 8 ? 3 : 2
      // batchSize should round up.
      let batchSize = channels == 8 ? (xT.shape[0] + 2) / 3 : (xT.shape[0] + 1) / 2
      let batch: MLArrayBatchProvider
      let conditionalLength = c.shape[2]
      let startHeight = xT.shape[1]
      let startWidth = xT.shape[2]
      if batchSize == 1 {
        var xTSlice = xT.permuted(0, 3, 1, 2).rawValue.copied().toCPU()
        var hiddenStates = DynamicGraph.Tensor<FloatType>(c).transposed(1, 2).reshaped(
          .NHWC(c.shape[0], conditionalLength, 1, 77)
        )
        .rawValue.toCPU()
        if xT.shape[0] != insideBatch {
          let oldXTSlice = xTSlice
          xTSlice = Tensor<FloatType>(.CPU, .NCHW(insideBatch, channels, startHeight, startWidth))
          for i in 0..<insideBatch {
            xTSlice[i..<(i + 1), 0..<channels, 0..<startHeight, 0..<startWidth] =
              oldXTSlice[0..<1, 0..<channels, 0..<startHeight, 0..<startWidth]
          }
        }
        if c.shape[0] != insideBatch {
          let oldHiddenStates = hiddenStates
          hiddenStates = Tensor<FloatType>(.CPU, .NHWC(insideBatch, conditionalLength, 1, 77))
          for i in 0..<insideBatch {
            hiddenStates[i..<(i + 1), 0..<conditionalLength, 0..<1, 0..<77] =
              oldHiddenStates[0..<1, 0..<conditionalLength, 0..<1, 0..<77]
          }
        }
        let inputs = try! MLDictionaryFeatureProvider(dictionary: [
          "sample": MLMultiArray(
            MLShapedArray<Float>(
              Tensor<Float>(from: xTSlice))
          ),
          "timestep": MLMultiArray(
            MLShapedArray<Float>(
              scalars: Array(repeating: timestep.now, count: insideBatch), shape: [insideBatch])),
          "encoder_hidden_states": MLMultiArray(
            MLShapedArray<Float>(
              Tensor<Float>(
                from: hiddenStates))
          ),
        ])
        batch = MLArrayBatchProvider(array: [inputs])
      } else {
        let xTtensor = xT.permuted(0, 3, 1, 2).rawValue.copied().toCPU()
        let cTensor = DynamicGraph.Tensor<FloatType>(c).transposed(1, 2).reshaped(
          .NHWC(c.shape[0], conditionalLength, 1, 77)
        )
        .rawValue.toCPU()
        var inputs = [MLDictionaryFeatureProvider]()
        for i in 0..<batchSize {
          var slice = Tensor<FloatType>(.CPU, .NCHW(insideBatch, channels, startHeight, startWidth))  // Fixed, for now.
          var hiddenStates = Tensor<FloatType>(.CPU, .NHWC(insideBatch, conditionalLength, 1, 77))  // Fixeed, for now.
          for j in 0..<insideBatch {
            let sliceStart = i + batchSize * j
            let xTSliceStart = min(sliceStart, xTtensor.shape[0] - 1)
            let cSliceStart = min(sliceStart, cTensor.shape[0] - 1)
            slice[j..<(j + 1), 0..<channels, 0..<startHeight, 0..<startWidth] =
              xTtensor[
                xTSliceStart..<(xTSliceStart + 1), 0..<channels, 0..<startHeight,
                0..<startWidth]
            hiddenStates[j..<(j + 1), 0..<conditionalLength, 0..<1, 0..<77] =
              cTensor[
                cSliceStart..<(cSliceStart + 1), 0..<conditionalLength, 0..<1, 0..<77]
          }
          inputs.append(
            try! MLDictionaryFeatureProvider(dictionary: [
              "sample": MLMultiArray(
                MLShapedArray<Float>(
                  Tensor<Float>(from: slice))
              ),
              "timestep": MLMultiArray(
                MLShapedArray<Float>(
                  scalars: Array(repeating: timestep.now, count: insideBatch), shape: [insideBatch])
              ),
              "encoder_hidden_states": MLMultiArray(
                MLShapedArray<Float>(
                  Tensor<Float>(from: hiddenStates))
              ),
            ]))
        }
        batch = MLArrayBatchProvider(array: inputs)
      }
      let results = try! unetChunk1.perform { try $0.predictions(fromBatch: batch) }
      let inputsForResults = batch.arrayOfFeatureValueDictionaries
      let next = try! results.arrayOfFeatureValueDictionaries.enumerated().map { (index, dict) in
        let nextDict = dict.merging(inputsForResults[index]) { (out, _) in out }
        return try MLDictionaryFeatureProvider(dictionary: nextDict)
      }
      let nextBatch = MLArrayBatchProvider(array: next)
      if reduceMemory {
        unetChunk1.unloadResources()
      }
      let resultBatch = (try! unetChunk2.perform { try $0.predictions(fromBatch: nextBatch) })
        .arrayOfFeatureValueDictionaries
      let et: Tensor<FloatType>
      if batchSize == 1 {
        let noisePred = resultBatch[0]["noise_pred"]!
        et = Tensor<FloatType>(from: Tensor(noisePred.shapedArrayValue(of: Float.self)!))
          .permuted(
            0, 2, 3, 1
          ).copied().reshaped(format: xT.format, shape: xT.shape).toGPU()
      } else {
        var etc = Tensor<FloatType>(
          .CPU, .NHWC(xT.shape[0], startHeight, startWidth, 4))
        for i in 0..<batchSize {
          let noisePred = resultBatch[i]["noise_pred"]!
          let slice = Tensor<FloatType>(from: Tensor(noisePred.shapedArrayValue(of: Float.self)!))
            .permuted(
              0, 2, 3, 1
            ).copied()
          for j in 0..<insideBatch {
            let sliceStart = i + batchSize * j
            guard sliceStart < etc.shape[0] else { break }
            etc[
              sliceStart..<(sliceStart + 1), 0..<startHeight, 0..<startWidth, 0..<4] =
              slice[j..<(j + 1), 0..<startHeight, 0..<startWidth, 0..<4]
          }
        }
        et = etc.toGPU(0)
      }
      if reduceMemory {
        unetChunk2.unloadResources()
      }
      return graph.variable(et)
    }
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    return x
  }

  public mutating func cancel() {
    // Do nothing.
  }
}

extension MLFeatureProvider {
  var featureValueDictionary: [String: MLFeatureValue] {
    self.featureNames.reduce(into: [String: MLFeatureValue]()) { result, name in
      result[name] = self.featureValue(for: name)
    }
  }
}

extension MLBatchProvider {
  var arrayOfFeatureValueDictionaries: [[String: MLFeatureValue]] {
    (0..<self.count).map {
      self.features(at: $0).featureValueDictionary
    }
  }
}
