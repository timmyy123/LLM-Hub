public enum ModelWeightFormat {
  case diffusers
  case generativeModels
}

public struct ModelWeightElement: RandomAccessCollection, ExpressibleByArrayLiteral {
  public enum Format {
    case O
    case I
  }
  public let format: Format
  public let offsets: [Int]?
  public let scale: Float
  public let shift: Float
  public let index: Int
  public let isBF16: Bool
  public let interleavedIndices: Set<Int>
  public let numberOfHeads: Int
  public let headDimension: Int
  // A prefix of each head can be interleaved while its remaining channels stay unchanged.
  public let interleavedDimension: Int?
  private let underlyingArray: [String]

  public typealias Element = String
  public typealias Index = Int
  public typealias Indices = Range<Index>
  public typealias SubSequence = Array<Element>.SubSequence
  public var endIndex: Index { underlyingArray.endIndex }
  public var indices: Indices { underlyingArray.indices }
  public var startIndex: Index { underlyingArray.startIndex }
  public func formIndex(after i: inout Index) { underlyingArray.formIndex(after: &i) }
  public func formIndex(before i: inout Index) { underlyingArray.formIndex(before: &i) }
  public subscript(position: Index) -> Element { underlyingArray[position] }
  public subscript(x: Indices) -> SubSequence { underlyingArray[x] }

  public init(
    _ array: [Element], format: Format = .O, offsets: [Int]? = nil, scale: Float = 1,
    shift: Float = 0,
    index: Int = 0, isBF16: Bool = false, interleavedIndices: Set<Int> = [],
    numberOfHeads: Int = 0, headDimension: Int = 0, interleavedDimension: Int? = nil
  ) {
    self.underlyingArray = array
    self.format = format
    self.offsets = offsets
    self.scale = scale
    self.shift = shift
    self.index = index
    self.isBF16 = isBF16
    self.interleavedIndices = interleavedIndices
    self.numberOfHeads = numberOfHeads
    self.headDimension = headDimension
    self.interleavedDimension = interleavedDimension
  }

  public init(arrayLiteral elements: Element...) {
    self.underlyingArray = elements
    self.format = .O
    self.offsets = nil
    self.scale = 1
    self.shift = 0
    self.index = 0
    self.isBF16 = false
    self.interleavedIndices = []
    self.numberOfHeads = 0
    self.headDimension = 0
    self.interleavedDimension = nil
  }
}

public typealias ModelWeightMapping = [String: ModelWeightElement]

public typealias ModelWeightMapper = (ModelWeightFormat) -> ModelWeightMapping
