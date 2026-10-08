import Foundation

#if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
  public typealias FloatType = Float16
#else
  public typealias FloatType = Float
#endif
