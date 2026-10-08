extension Array {
  public func sortedWith<T: Comparable>(_ closure: (Element) -> (T)) -> [Element] {
    return sorted { a, b in
      closure(a) < closure(b)
    }
  }
}
