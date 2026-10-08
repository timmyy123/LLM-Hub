import Foundation

extension String: Error {}

extension String {
  public func removingSuffix(_ suffix: String) -> String {
    if hasSuffix(suffix) {
      var copy = String(self)
      let range = copy.range(of: suffix, options: .backwards)!
      copy.removeSubrange(range)
      return copy
    } else {
      return self
    }
  }
}

extension Character {
  public var isHan: Bool {
    for scalar in unicodeScalars {
      switch scalar.value {
      case 19968...40959:  // Common
        return true
      case 13312...19903:  // Rare
        return true
      case 131072...173791:  // Rare, historic
        return true
      case 173824...177983:  // Rare, historic
        return true
      case 177984...178207:  // Uncommon
        return true
      case 63744...64255:  // Duplicates
        return true
      case 194560...195103:  // Unifiable variants
        return true
      case 40870...40883:  // Interoperability with HKSCS standard
        return true
      case 40884...40891:  // Interoperability with GB 18030 standard
        return true
      case 40892...40898:  // Interoperability with commercial implementations
        return true
      case 40899:  // Correction of mistaken unification
        return true
      case 40900...40902:  // Interoperability with ARIB standard
        return true
      case 40903...40907:  // Interoperability with HKSCS standard
        return true
      default:
        continue
      }
    }
    return false
  }
}
