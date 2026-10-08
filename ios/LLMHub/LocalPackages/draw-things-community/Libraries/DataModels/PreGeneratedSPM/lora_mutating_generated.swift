import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

// MARK - SQLiteValue for Enumerations

extension LoRATrainableLayer: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

// MARK - Serializer

extension LoRATrainingConfiguration: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __name = self.name.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __baseModel = self.baseModel.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __triggerWord = self.triggerWord.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __autoFillPrompt =
      self.autoFillPrompt.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    var __trainableLayers = [zzz_DflatGen_LoRATrainableLayer]()
    for i in self.trainableLayers {
      __trainableLayers.append(
        zzz_DflatGen_LoRATrainableLayer(rawValue: i.rawValue) ?? .latentsembedder)
    }
    let __vector_trainableLayers = flatBufferBuilder.createVector(__trainableLayers)
    let __vector_layerIndices = flatBufferBuilder.createVector(self.layerIndices)
    let __vector_additionalScales = flatBufferBuilder.createVector(self.additionalScales)
    let start = zzz_DflatGen_LoRATrainingConfiguration.startLoRATrainingConfiguration(
      &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(id: self.id, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(name: __name, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(startWidth: self.startWidth, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(startHeight: self.startHeight, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(seed: self.seed, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      trainingSteps: self.trainingSteps, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(baseModel: __baseModel, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(networkDim: self.networkDim, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(networkScale: self.networkScale, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      unetLearningRate: self.unetLearningRate, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      saveEveryNSteps: self.saveEveryNSteps, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(warmupSteps: self.warmupSteps, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      gradientAccumulationSteps: self.gradientAccumulationSteps, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      cotrainTextModel: self.cotrainTextModel, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      textModelLearningRate: self.textModelLearningRate, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(clipSkip: self.clipSkip, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(noiseOffset: self.noiseOffset, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      denoisingStart: self.denoisingStart, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(denoisingEnd: self.denoisingEnd, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(triggerWord: __triggerWord, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(autoFillPrompt: __autoFillPrompt, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      autoCaptioning: self.autoCaptioning, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      cotrainCustomEmbedding: self.cotrainCustomEmbedding, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      customEmbeddingLearningRate: self.customEmbeddingLearningRate, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      customEmbeddingLength: self.customEmbeddingLength, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      stopEmbeddingTrainingAtStep: self.stopEmbeddingTrainingAtStep, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.addVectorOf(
      trainableLayers: __vector_trainableLayers, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.addVectorOf(
      layerIndices: __vector_layerIndices, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(shift: self.shift, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      resolutionDependentShift: self.resolutionDependentShift, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      guidanceEmbedLowerBound: self.guidanceEmbedLowerBound, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      guidanceEmbedUpperBound: self.guidanceEmbedUpperBound, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      unetLearningRateLowerBound: self.unetLearningRateLowerBound, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      stepsBetweenRestarts: self.stepsBetweenRestarts, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      captionDropoutRate: self.captionDropoutRate, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      orthonormalLoraDown: self.orthonormalLoraDown, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      maxTextLength: self.maxTextLength, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      useImageAspectRatio: self.useImageAspectRatio, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.addVectorOf(
      additionalScales: __vector_additionalScales, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      powerEmaLowerBound: self.powerEmaLowerBound, &flatBufferBuilder)
    zzz_DflatGen_LoRATrainingConfiguration.add(
      powerEmaUpperBound: self.powerEmaUpperBound, &flatBufferBuilder)
    return zzz_DflatGen_LoRATrainingConfiguration.endLoRATrainingConfiguration(
      &flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == LoRATrainingConfiguration {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension LoRATrainingConfiguration {
  public func toData() -> Data {
    var fbb = FlatBufferBuilder()
    let offset = to(flatBufferBuilder: &fbb)
    fbb.finish(offset: offset)
    return fbb.data
  }
}

// MARK - ChangeRequest

public final class LoRATrainingConfigurationChangeRequest: Dflat.ChangeRequest {
  private var _o: LoRATrainingConfiguration?
  public typealias Element = LoRATrainingConfiguration
  public var _type: ChangeRequestType
  public var _rowid: Int64
  public var id: Int64
  public var name: String?
  public var startWidth: UInt16
  public var startHeight: UInt16
  public var seed: UInt32
  public var trainingSteps: UInt32
  public var baseModel: String?
  public var networkDim: UInt16
  public var networkScale: Float32
  public var unetLearningRate: Float32
  public var saveEveryNSteps: UInt32
  public var warmupSteps: UInt32
  public var gradientAccumulationSteps: UInt32
  public var cotrainTextModel: Bool
  public var textModelLearningRate: Float32
  public var clipSkip: UInt32
  public var noiseOffset: Float32
  public var denoisingStart: Float32
  public var denoisingEnd: Float32
  public var triggerWord: String?
  public var autoFillPrompt: String?
  public var autoCaptioning: Bool
  public var cotrainCustomEmbedding: Bool
  public var customEmbeddingLearningRate: Float32
  public var customEmbeddingLength: UInt32
  public var stopEmbeddingTrainingAtStep: UInt32
  public var trainableLayers: [LoRATrainableLayer]
  public var layerIndices: [UInt32]
  public var shift: Float32
  public var resolutionDependentShift: Bool
  public var guidanceEmbedLowerBound: Float32
  public var guidanceEmbedUpperBound: Float32
  public var unetLearningRateLowerBound: Float32
  public var stepsBetweenRestarts: UInt32
  public var captionDropoutRate: Float32
  public var orthonormalLoraDown: Bool
  public var maxTextLength: UInt32
  public var useImageAspectRatio: Bool
  public var additionalScales: [UInt16]
  public var powerEmaLowerBound: Float32
  public var powerEmaUpperBound: Float32
  private init(type _type: ChangeRequestType) {
    _o = nil
    self._type = _type
    _rowid = -1
    id = 0
    name = nil
    startWidth = 0
    startHeight = 0
    seed = 0
    trainingSteps = 0
    baseModel = nil
    networkDim = 0
    networkScale = 0.0
    unetLearningRate = 0.0
    saveEveryNSteps = 0
    warmupSteps = 0
    gradientAccumulationSteps = 0
    cotrainTextModel = false
    textModelLearningRate = 0.0
    clipSkip = 1
    noiseOffset = 0.0
    denoisingStart = 0.0
    denoisingEnd = 0.0
    triggerWord = nil
    autoFillPrompt = nil
    autoCaptioning = false
    cotrainCustomEmbedding = false
    customEmbeddingLearningRate = 0.05
    customEmbeddingLength = 4
    stopEmbeddingTrainingAtStep = 500
    trainableLayers = []
    layerIndices = []
    shift = 1.0
    resolutionDependentShift = false
    guidanceEmbedLowerBound = 3.0
    guidanceEmbedUpperBound = 4.0
    unetLearningRateLowerBound = 0.0
    stepsBetweenRestarts = 200
    captionDropoutRate = 0.0
    orthonormalLoraDown = false
    maxTextLength = 512
    useImageAspectRatio = false
    additionalScales = []
    powerEmaLowerBound = 0.0
    powerEmaUpperBound = 0.0
  }
  private init(type _type: ChangeRequestType, _ _o: LoRATrainingConfiguration) {
    self._o = _o
    self._type = _type
    _rowid = _o._rowid
    id = _o.id
    name = _o.name
    startWidth = _o.startWidth
    startHeight = _o.startHeight
    seed = _o.seed
    trainingSteps = _o.trainingSteps
    baseModel = _o.baseModel
    networkDim = _o.networkDim
    networkScale = _o.networkScale
    unetLearningRate = _o.unetLearningRate
    saveEveryNSteps = _o.saveEveryNSteps
    warmupSteps = _o.warmupSteps
    gradientAccumulationSteps = _o.gradientAccumulationSteps
    cotrainTextModel = _o.cotrainTextModel
    textModelLearningRate = _o.textModelLearningRate
    clipSkip = _o.clipSkip
    noiseOffset = _o.noiseOffset
    denoisingStart = _o.denoisingStart
    denoisingEnd = _o.denoisingEnd
    triggerWord = _o.triggerWord
    autoFillPrompt = _o.autoFillPrompt
    autoCaptioning = _o.autoCaptioning
    cotrainCustomEmbedding = _o.cotrainCustomEmbedding
    customEmbeddingLearningRate = _o.customEmbeddingLearningRate
    customEmbeddingLength = _o.customEmbeddingLength
    stopEmbeddingTrainingAtStep = _o.stopEmbeddingTrainingAtStep
    trainableLayers = _o.trainableLayers
    layerIndices = _o.layerIndices
    shift = _o.shift
    resolutionDependentShift = _o.resolutionDependentShift
    guidanceEmbedLowerBound = _o.guidanceEmbedLowerBound
    guidanceEmbedUpperBound = _o.guidanceEmbedUpperBound
    unetLearningRateLowerBound = _o.unetLearningRateLowerBound
    stepsBetweenRestarts = _o.stepsBetweenRestarts
    captionDropoutRate = _o.captionDropoutRate
    orthonormalLoraDown = _o.orthonormalLoraDown
    maxTextLength = _o.maxTextLength
    useImageAspectRatio = _o.useImageAspectRatio
    additionalScales = _o.additionalScales
    powerEmaLowerBound = _o.powerEmaLowerBound
    powerEmaUpperBound = _o.powerEmaUpperBound
  }
  public static func changeRequest(_ o: LoRATrainingConfiguration)
    -> LoRATrainingConfigurationChangeRequest?
  {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.id])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: LoRATrainingConfiguration.self, for: key)
    return u.map { LoRATrainingConfigurationChangeRequest(type: .update, $0) }
  }
  public static func upsertRequest(_ o: LoRATrainingConfiguration)
    -> LoRATrainingConfigurationChangeRequest
  {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.id])
    guard
      let u = transactionContext.objectRepository.object(
        transactionContext.connection, ofType: LoRATrainingConfiguration.self, for: key)
    else {
      return Self.creationRequest(o)
    }
    let changeRequest = LoRATrainingConfigurationChangeRequest(type: .update, o)
    changeRequest._o = u
    changeRequest._rowid = u._rowid
    return changeRequest
  }
  public static func creationRequest(_ o: LoRATrainingConfiguration)
    -> LoRATrainingConfigurationChangeRequest
  {
    let creationRequest = LoRATrainingConfigurationChangeRequest(type: .creation, o)
    creationRequest._rowid = -1
    return creationRequest
  }
  public static func creationRequest() -> LoRATrainingConfigurationChangeRequest {
    return LoRATrainingConfigurationChangeRequest(type: .creation)
  }
  public static func deletionRequest(_ o: LoRATrainingConfiguration)
    -> LoRATrainingConfigurationChangeRequest?
  {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.id])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: LoRATrainingConfiguration.self, for: key)
    return u.map { LoRATrainingConfigurationChangeRequest(type: .deletion, $0) }
  }
  var _atom: LoRATrainingConfiguration {
    let atom = LoRATrainingConfiguration(
      id: id, name: name, startWidth: startWidth, startHeight: startHeight, seed: seed,
      trainingSteps: trainingSteps, baseModel: baseModel, networkDim: networkDim,
      networkScale: networkScale, unetLearningRate: unetLearningRate,
      saveEveryNSteps: saveEveryNSteps, warmupSteps: warmupSteps,
      gradientAccumulationSteps: gradientAccumulationSteps, cotrainTextModel: cotrainTextModel,
      textModelLearningRate: textModelLearningRate, clipSkip: clipSkip, noiseOffset: noiseOffset,
      denoisingStart: denoisingStart, denoisingEnd: denoisingEnd, triggerWord: triggerWord,
      autoFillPrompt: autoFillPrompt, autoCaptioning: autoCaptioning,
      cotrainCustomEmbedding: cotrainCustomEmbedding,
      customEmbeddingLearningRate: customEmbeddingLearningRate,
      customEmbeddingLength: customEmbeddingLength,
      stopEmbeddingTrainingAtStep: stopEmbeddingTrainingAtStep, trainableLayers: trainableLayers,
      layerIndices: layerIndices, shift: shift, resolutionDependentShift: resolutionDependentShift,
      guidanceEmbedLowerBound: guidanceEmbedLowerBound,
      guidanceEmbedUpperBound: guidanceEmbedUpperBound,
      unetLearningRateLowerBound: unetLearningRateLowerBound,
      stepsBetweenRestarts: stepsBetweenRestarts, captionDropoutRate: captionDropoutRate,
      orthonormalLoraDown: orthonormalLoraDown, maxTextLength: maxTextLength,
      useImageAspectRatio: useImageAspectRatio, additionalScales: additionalScales,
      powerEmaLowerBound: powerEmaLowerBound, powerEmaUpperBound: powerEmaUpperBound)
    atom._rowid = _rowid
    return atom
  }
  public func commit(_ toolbox: PersistenceToolbox) -> UpdatedObject? {
    guard let toolbox = toolbox as? SQLitePersistenceToolbox else { return nil }
    switch _type {
    case .creation:
      guard
        let insert = toolbox.connection.prepareStaticStatement(
          "INSERT INTO loratrainingconfiguration (__pk0, p) VALUES (?1, ?2)")
      else { return nil }
      id.bindSQLite(insert, parameterId: 1)
      let atom = self._atom
      toolbox.flatBufferBuilder.clear()
      let offset = atom.to(flatBufferBuilder: &toolbox.flatBufferBuilder)
      toolbox.flatBufferBuilder.finish(offset: offset)
      let byteBuffer = toolbox.flatBufferBuilder.buffer
      let memory = byteBuffer.memory.advanced(by: byteBuffer.reader)
      let SQLITE_STATIC = unsafeBitCast(
        OpaquePointer(bitPattern: 0), to: sqlite3_destructor_type.self)
      sqlite3_bind_blob(insert, 2, memory, Int32(byteBuffer.size), SQLITE_STATIC)
      guard SQLITE_DONE == sqlite3_step(insert) else { return nil }
      _rowid = sqlite3_last_insert_rowid(toolbox.connection.sqlite)
      _type = .none
      atom._rowid = _rowid
      return .inserted(atom)
    case .update:
      guard let o = _o else { return nil }
      let atom = self._atom
      guard atom != o else {
        _type = .none
        return .identity(atom)
      }
      guard
        let update = toolbox.connection.prepareStaticStatement(
          "REPLACE INTO loratrainingconfiguration (__pk0, p, rowid) VALUES (?1, ?2, ?3)")
      else { return nil }
      id.bindSQLite(update, parameterId: 1)
      toolbox.flatBufferBuilder.clear()
      let offset = atom.to(flatBufferBuilder: &toolbox.flatBufferBuilder)
      toolbox.flatBufferBuilder.finish(offset: offset)
      let byteBuffer = toolbox.flatBufferBuilder.buffer
      let memory = byteBuffer.memory.advanced(by: byteBuffer.reader)
      let SQLITE_STATIC = unsafeBitCast(
        OpaquePointer(bitPattern: 0), to: sqlite3_destructor_type.self)
      sqlite3_bind_blob(update, 2, memory, Int32(byteBuffer.size), SQLITE_STATIC)
      _rowid.bindSQLite(update, parameterId: 3)
      guard SQLITE_DONE == sqlite3_step(update) else { return nil }
      _type = .none
      return .updated(atom)
    case .deletion:
      guard
        let deletion = toolbox.connection.prepareStaticStatement(
          "DELETE FROM loratrainingconfiguration WHERE rowid=?1")
      else { return nil }
      _rowid.bindSQLite(deletion, parameterId: 1)
      guard SQLITE_DONE == sqlite3_step(deletion) else { return nil }
      _type = .none
      return .deleted(_rowid)
    case .none:
      preconditionFailure()
    }
  }
}
