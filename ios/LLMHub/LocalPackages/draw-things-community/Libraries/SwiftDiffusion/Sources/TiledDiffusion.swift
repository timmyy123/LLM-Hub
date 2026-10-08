public struct TiledConfiguration: Equatable {
  public struct Size: Equatable {
    public var width: Int
    public var height: Int
    public init(width: Int, height: Int) {
      self.width = width
      self.height = height
    }
  }
  public var isEnabled: Bool
  public var tileSize: Size
  public var tileOverlap: Int
  public init(isEnabled: Bool, tileSize: Size, tileOverlap: Int) {
    self.isEnabled = isEnabled
    self.tileSize = tileSize
    self.tileOverlap = tileOverlap
  }
}

func paddedTileStartAndEnd(iOfs: Int, length: Int, tileSize: Int, tileOverlap: Int) -> (
  paddedStart: Int, paddedEnd: Int
) {
  let inputEnd = min(iOfs + tileSize - tileOverlap * 2, length)
  var inputStartPad = iOfs - tileOverlap
  var inputEndPad = inputEnd + tileOverlap
  if inputStartPad < 0 {
    inputStartPad = 0
    inputEndPad = tileSize
    precondition(inputEndPad <= length)
  } else if inputEndPad >= length {
    inputEndPad = length
    inputStartPad = length - tileSize
    precondition(inputStartPad >= 0)
  }
  return (inputStartPad, inputEndPad)
}

func xyTileWeightsAndIndexes(
  width: Int, height: Int, xTiles: Int, yTiles: Int, tileSize: (width: Int, height: Int),
  tileOverlap: Int
) -> ([[(weight: Float, index: Int, offset: Int)]], [[(weight: Float, index: Int, offset: Int)]]) {
  var yWeightsAndIndexes = [[(weight: Float, index: Int, offset: Int)]]()
  for j in 0..<height {
    var weightAndIndex = [(weight: Float, index: Int, offset: Int)]()
    let y1 = min(
      max((j - tileOverlap) / ((tileSize.height - tileOverlap * 2)), 0), yTiles - 1)
    let y1Ofs = y1 * (tileSize.height - tileOverlap * 2) + (y1 > 0 ? tileOverlap : 0)
    let (inputStartY1Pad, inputEndY1Pad) = paddedTileStartAndEnd(
      iOfs: y1Ofs, length: height, tileSize: tileSize.height,
      tileOverlap: tileOverlap)
    if j >= inputStartY1Pad && j < inputEndY1Pad {
      weightAndIndex.append(
        (
          weight: Float(min(j - inputStartY1Pad, inputEndY1Pad - j)), index: y1,
          offset: j - inputStartY1Pad
        ))
    }
    if y1 + 1 < yTiles {
      let y2Ofs = (y1 + 1) * (tileSize.height - tileOverlap * 2) + tileOverlap
      let (inputStartY2Pad, inputEndY2Pad) = paddedTileStartAndEnd(
        iOfs: y2Ofs, length: height, tileSize: tileSize.height,
        tileOverlap: tileOverlap)
      if j >= inputStartY2Pad && j < inputEndY2Pad {
        weightAndIndex.append(
          (
            weight: Float(min(j - inputStartY2Pad, inputEndY2Pad - j)), index: y1 + 1,
            offset: j - inputStartY2Pad
          ))
      }
    }
    if y1 - 1 >= 0 {
      let y0Ofs =
        (y1 - 1) * (tileSize.height - tileOverlap * 2) + (y1 - 1 > 0 ? tileOverlap : 0)
      let (inputStartY0Pad, inputEndY0Pad) = paddedTileStartAndEnd(
        iOfs: y0Ofs, length: height, tileSize: tileSize.height,
        tileOverlap: tileOverlap)
      if j >= inputStartY0Pad && j < inputEndY0Pad {
        weightAndIndex.append(
          (
            weight: Float(min(j - inputStartY0Pad, inputEndY0Pad - j)), index: y1 - 1,
            offset: j - inputStartY0Pad
          ))
      }
    }
    // Now normalize the weights.
    let totalWeight = weightAndIndex.reduce(0) { $0 + $1.weight }
    yWeightsAndIndexes.append(
      weightAndIndex.map {
        if totalWeight > 0 {  // Fix boundary condition.
          return (weight: $0.weight / totalWeight, index: $0.index, offset: $0.offset)
        } else {
          return (weight: 1, index: $0.index, offset: $0.offset)
        }
      })
  }
  var xWeightsAndIndexes = [[(weight: Float, index: Int, offset: Int)]]()
  for i in 0..<width {
    var weightAndIndex = [(weight: Float, index: Int, offset: Int)]()
    let x1 = min(
      max((i - tileOverlap) / ((tileSize.width - tileOverlap * 2)), 0), xTiles - 1)
    let x1Ofs = x1 * (tileSize.width - tileOverlap * 2) + (x1 > 0 ? tileOverlap : 0)
    let (inputStartX1Pad, inputEndX1Pad) = paddedTileStartAndEnd(
      iOfs: x1Ofs, length: width, tileSize: tileSize.width,
      tileOverlap: tileOverlap)
    if i >= inputStartX1Pad && i < inputEndX1Pad {
      weightAndIndex.append(
        (
          weight: Float(min(i - inputStartX1Pad, inputEndX1Pad - i)), index: x1,
          offset: i - inputStartX1Pad
        ))
    }
    if x1 + 1 < xTiles {
      let x2Ofs = (x1 + 1) * (tileSize.width - tileOverlap * 2) + tileOverlap
      let (inputStartX2Pad, inputEndX2Pad) = paddedTileStartAndEnd(
        iOfs: x2Ofs, length: width, tileSize: tileSize.width,
        tileOverlap: tileOverlap)
      if i >= inputStartX2Pad && i < inputEndX2Pad {
        weightAndIndex.append(
          (
            weight: Float(min(i - inputStartX2Pad, inputEndX2Pad - i)), index: x1 + 1,
            offset: i - inputStartX2Pad
          ))
      }
    }
    if x1 - 1 >= 0 {
      let x0Ofs =
        (x1 - 1) * (tileSize.width - tileOverlap * 2) + (x1 - 1 > 0 ? tileOverlap : 0)
      let (inputStartX0Pad, inputEndX0Pad) = paddedTileStartAndEnd(
        iOfs: x0Ofs, length: width, tileSize: tileSize.width,
        tileOverlap: tileOverlap)
      if i >= inputStartX0Pad && i < inputEndX0Pad {
        weightAndIndex.append(
          (
            weight: Float(min(i - inputStartX0Pad, inputEndX0Pad - i)), index: x1 - 1,
            offset: i - inputStartX0Pad
          ))
      }
    }
    // Now normalize the weights.
    let totalWeight = weightAndIndex.reduce(0) { $0 + $1.weight }
    xWeightsAndIndexes.append(
      weightAndIndex.map {
        if totalWeight > 0 {  // Fix boundary condition.
          return (weight: $0.weight / totalWeight, index: $0.index, offset: $0.offset)
        } else {
          return (weight: 1, index: $0.index, offset: $0.offset)
        }
      })
  }
  return (xWeightsAndIndexes, yWeightsAndIndexes)
}
