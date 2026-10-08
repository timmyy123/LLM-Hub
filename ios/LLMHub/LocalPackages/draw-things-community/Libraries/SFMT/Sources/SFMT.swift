import C_sfmt

public struct SFMT: RandomNumberGenerator {
  private var state: sfmt_t
  public init(seed: UInt64) {
    state = sfmt_t()
    sfmt_init_gen_rand(&state, UInt32(truncatingIfNeeded: seed))
  }
  public mutating func next() -> UInt64 {
    return sfmt_genrand_uint64(&state)
  }
}
