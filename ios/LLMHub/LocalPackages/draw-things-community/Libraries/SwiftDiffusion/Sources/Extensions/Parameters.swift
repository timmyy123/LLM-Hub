import Fickling
import Foundation
import NNC
import ZIPFoundation

extension Model.Parameters {
  func copy<T: TensorNumeric>(
    from tensorDescriptor: TensorDescriptor, zip archive: TensorArchive, of type: T.Type
  ) throws {
    try archive.with(tensorDescriptor) {
      copy(from: Tensor<T>(from: $0))
    }
  }
}
