import Foundation

public enum Denoiser {
  public enum Conditioning: String, Codable {
    case noise
    case timestep
  }

  public enum Objective: Codable, Equatable {
    case edm(sigmaData: Double)  // EDM objective carries sigmaData, which is required for c_skip, c_out, c_in, c_noise computation.
    case v
    case epsilon
    case u(conditionScale: Double)  // Flow-Matching objective.
  }

  public enum Parameterization {
    case rf(RF)  // Rectified Flow.
    case edm(EDM)
    case ddpm(DDPM)
    public struct RF: Codable {
      public var sigmaMin: Double
      public var sigmaMax: Double
      public var conditionScale: Double
      public var shiftTerminal: Double?
      public init(
        sigmaMin: Double = 0, sigmaMax: Double = 1, conditionScale: Double = 1000,
        shiftTerminal: Double? = nil
      ) {
        self.sigmaMin = sigmaMin
        self.sigmaMax = sigmaMax
        self.conditionScale = conditionScale
        self.shiftTerminal = shiftTerminal
      }
    }
    public struct EDM: Codable {
      public var sigmaMin: Double
      public var sigmaMax: Double
      public var sigmaData: Double
      public init(sigmaMin: Double = 0.002, sigmaMax: Double = 80.0, sigmaData: Double = 0.5) {
        self.sigmaMin = sigmaMin
        self.sigmaMax = sigmaMax
        self.sigmaData = sigmaData
      }
    }
    public struct DDPM: Codable {
      public enum Linspace: String, Codable {
        case linearWrtSigma
        case linearWrtBeta
      }
      public var linearStart: Double
      public var linearEnd: Double
      public var timesteps: Int
      public var linspace: Linspace
      public init(linearStart: Double, linearEnd: Double, timesteps: Int, linspace: Linspace) {
        self.linearStart = linearStart
        self.linearEnd = linearEnd
        self.timesteps = timesteps
        self.linspace = linspace
      }
      private var betas: [Double] {  // Linear for now.
        var betas = [Double]()
        let start = Double(linspace == .linearWrtSigma ? linearStart.squareRoot() : linearStart)
        let length =
          Double(linspace == .linearWrtSigma ? linearEnd.squareRoot() : linearEnd) - start
        for i in 0..<timesteps {
          let beta = start + Double(i) * length / Double(timesteps - 1)
          switch linspace {
          case .linearWrtSigma:
            betas.append(beta * beta)
          case .linearWrtBeta:
            betas.append(beta)
          }
        }
        return betas
      }
      public var alphasCumprod: [Double] {
        var cumprod: Double = 1
        return betas.map {
          cumprod *= 1 - $0
          return cumprod
        }
      }
    }
  }
}

public protocol DenoiserDiscretization {
  // Conditioned "timestep". For DDPM, these are real timesteps, for continuous space, this is between 0 and 1.
  func timestep(for alphaCumprod: Double) -> Float
  func alphaCumprod(timestep: Float, shift: Double) -> Double
  func alphasCumprod(steps: Int, shift: Double) -> [Double]
  var objective: Denoiser.Objective { get }
  // For continuous space, this is 1, for DDPM, this is number of steps during training.
  var timesteps: Float { get }
  func sigma(from alphaCumprod: Double) -> Double
  func alphaCumprod(from sigma: Double) -> Double
}

extension DenoiserDiscretization {
  public func alphasCumprod(steps: Int) -> [Double] {
    alphasCumprod(steps: steps, shift: 1)
  }

  // Conditioned on "noise". This is different depending on the objective.
  func noise(for alphaCumprod: Double) -> Float {
    let sigma = ((1 - alphaCumprod) / alphaCumprod).squareRoot()
    switch objective {
    case .edm(let sigmaData):
      return Float((sigmaData * sigmaData) * log(sigma))
    case .v:
      return Float(0.25 * log(sigma))
    case .epsilon:
      return Float(sigma)
    case .u(let conditionScale):
      return Float(conditionScale * alphaCumprod)
    }
  }
}

extension Denoiser {
  public typealias Discretization = DenoiserDiscretization
}

extension Denoiser {
  public struct CosineDiscretization: Discretization {
    private let s: Double
    private let range: ClosedRange<Double>
    private let minVar: Double
    private let sigmas: [Double]?
    private let rf: Bool
    public let objective: Objective
    public let timesteps: Float
    init(
      objective: Objective, timesteps: Float, s: Double = 0.008,
      range: ClosedRange<Double> = 0.0001...0.9999, sigmas: [Double]? = nil,
      rf: Bool = false
    ) {
      self.objective = objective
      self.timesteps = timesteps
      self.s = s
      self.range = range
      self.sigmas = sigmas
      let minStd = cos(s / (1 + s) * Double.pi * 0.5)
      self.minVar = minStd * minStd
      self.rf = rf
    }
    public init(_ parameterization: Parameterization, objective: Objective, s: Double = 0.008) {
      switch parameterization {
      case .edm(let edm):
        let range: ClosedRange<Double> =
          (1.0 / (edm.sigmaMax * edm.sigmaMax + 1))...(1.0 / (edm.sigmaMin * edm.sigmaMin + 1))
        self.init(objective: objective, timesteps: 1, s: s, range: range)
      case .ddpm(let ddpm):
        let alphasCumprod = ddpm.alphasCumprod
        let sigmas = alphasCumprod.map { ((1 - $0) / $0).squareRoot() }
        self.init(
          objective: objective, timesteps: Float(ddpm.timesteps), s: s,
          range: alphasCumprod[alphasCumprod.count - 1]...alphasCumprod[0], sigmas: sigmas)
      case .rf(let rf):
        self.init(
          objective: objective, timesteps: Float(rf.conditionScale), s: s,
          range: rf.sigmaMin...rf.sigmaMax, rf: true)
      }
    }
    public func timestep(for alphaCumprod: Double) -> Float {
      if let sigmas = sigmas {
        // This is for legacy DDPM.
        return timestep(for: alphaCumprod, in: sigmas)
      }
      let t = acos((alphaCumprod * minVar).squareRoot()) / (Double.pi * 0.5) * (1 + s) - s
      return Float(t) * timesteps
    }
    public func alphaCumprod(timestep: Float, shift: Double) -> Double {
      if let sigmas = sigmas {
        // This is for legacy DDPM.
        return alphaCumprod(timestep: timestep, in: sigmas)
      }
      let t = Double(timestep) / Double(timesteps)
      let std = min(max(cos((s + t) / (1 + s) * Double.pi * 0.5), 0), 1)
      var v = min(max((std * std) / minVar, range.lowerBound), range.upperBound)
      if shift != 1 {
        // Simplify Sigmoid[Log[x / (1 - x)] + 2 * Log[1 / shift]]
        let shiftedV = 1.0 / (1.0 + shift * shift * (1.0 - v) / v)
        v = min(max(shiftedV, range.lowerBound), range.upperBound)
      }
      return v
    }
    public func alphasCumprod(steps: Int, shift: Double) -> [Double] {
      var alphasCumprod = [Double]()
      for i in 0..<steps {
        let t = Double(steps - i) / Double(steps)
        let std = min(max(cos((s + t) / (1 + s) * Double.pi * 0.5), 0), 1)
        var v = min(max((std * std) / minVar, range.lowerBound), range.upperBound)
        if shift != 1 {
          // Simplify Sigmoid[Log[x / (1 - x)] + 2 * Log[1 / shift]]
          let shiftedV = 1.0 / (1.0 + shift * shift * (1.0 - v) / v)
          v = min(max(shiftedV, range.lowerBound), range.upperBound)
        }
        alphasCumprod.append(v)
      }
      alphasCumprod.append(1.0)
      return alphasCumprod
    }
    public func sigma(from alphaCumprod: Double) -> Double {
      if rf {
        return 1 - alphaCumprod
      } else {
        return ((1 - alphaCumprod) / alphaCumprod).squareRoot()
      }
    }
    public func alphaCumprod(from sigma: Double) -> Double {
      if rf {
        return 1 - sigma
      } else {
        return 1.0 / (sigma * sigma + 1)
      }
    }
  }
}

extension Denoiser {
  public struct KarrasDiscretization: Discretization {
    private let rho: Double
    private let sigmaMin: Double
    private let sigmaMax: Double
    private let sigmas: [Double]?
    private let rf: Bool
    public let timesteps: Float
    public let objective: Objective
    init(
      objective: Objective, timesteps: Float, sigmaMin: Double, sigmaMax: Double, rho: Double = 7.0,
      sigmas: [Double]? = nil, rf: Bool = false
    ) {
      self.objective = objective
      self.timesteps = timesteps
      self.sigmaMin = sigmaMin
      self.sigmaMax = sigmaMax
      self.rho = rho
      self.sigmas = sigmas
      self.rf = rf
    }
    public init(_ parameterization: Parameterization, objective: Objective, rho: Double = 7.0) {
      switch parameterization {
      case .edm(let edm):
        self.init(
          objective: objective, timesteps: 1, sigmaMin: edm.sigmaMin, sigmaMax: edm.sigmaMax,
          rho: rho)
      case .ddpm(let ddpm):
        let alphasCumprod = ddpm.alphasCumprod
        let sigmas = alphasCumprod.map { ((1.0 - $0) / $0).squareRoot() }
        let sigmaMin = sigmas[1]
        let sigmaMax = sigmas[sigmas.count - 2]
        self.init(
          objective: objective, timesteps: Float(ddpm.timesteps), sigmaMin: sigmaMin,
          sigmaMax: sigmaMax, rho: rho, sigmas: sigmas)
      case .rf(let rf):
        self.init(
          objective: objective, timesteps: Float(rf.conditionScale), sigmaMin: rf.sigmaMin,
          sigmaMax: rf.sigmaMax, rho: rho, rf: true)
      }
    }
    public func timestep(for alphaCumprod: Double) -> Float {
      if let sigmas = sigmas {
        // This is for Legacy discrete DDPM. The timestep is scaled already.
        return timestep(for: alphaCumprod, in: sigmas)
      }
      let sigma: Double
      if rf {
        sigma = 1 - alphaCumprod
      } else {
        sigma = ((1 - alphaCumprod) / alphaCumprod).squareRoot()
      }
      let lowerBound = sigmaMin
      let upperBound = sigmaMax
      let minInvRho = pow(lowerBound, 1.0 / rho)
      let maxInvRho = pow(upperBound, 1.0 / rho)
      // Get back to maxInvRho + Double(i) * (minInvRho - maxInvRho) / Double(steps - 1)
      let t = 1.0 - (pow(sigma, 1.0 / rho) - maxInvRho) / (minInvRho - maxInvRho)
      return min(max(0, Float(t)), 1) * timesteps
    }
    public func alphaCumprod(timestep: Float, shift: Double) -> Double {
      if let sigmas = sigmas {
        // This is for Legacy discrete DDPM. The timestep is scaled already.
        return alphaCumprod(timestep: timestep, in: sigmas)
      }
      let lowerBound = sigmaMin
      let upperBound = sigmaMax
      let minInvRho = pow(lowerBound, 1.0 / rho)
      let maxInvRho = pow(upperBound, 1.0 / rho)
      var sigma = pow(
        maxInvRho + (Double(timesteps) - Double(timestep)) / Double(timesteps)
          * (minInvRho - maxInvRho),
        rho)
      if rf {
        if shift != 1 {
          sigma = shift * sigma / (1 + (shift - 1) * sigma)
        }
        return 1 - sigma
      } else {
        if shift != 1 {
          sigma = shift * sigma
        }
        return (1 / (sigma * sigma + 1))
      }
    }
    public func alphasCumprod(steps: Int, shift: Double) -> [Double] {
      var alphasCumprod = [Double]()
      let lowerBound = sigmaMin
      let upperBound = sigmaMax
      let minInvRho = pow(lowerBound, 1.0 / rho)
      let maxInvRho = pow(upperBound, 1.0 / rho)
      for i in 0..<steps {
        if rf {
          var sigma = pow(maxInvRho + Double(i) * (minInvRho - maxInvRho) / Double(steps), rho)
          if shift != 1 {
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
          }
          alphasCumprod.append(1 - sigma)
        } else {
          var sigma = pow(maxInvRho + Double(i) * (minInvRho - maxInvRho) / Double(steps - 1), rho)
          if shift != 1 {
            // Simplify Sigmoid[Log[x / (1 - x)] + 2 * Log[1 / shift]]
            // var v: Double = 1.0 / (sigma * sigma + 1)
            // v = 1.0 / (1.0 + shift * shift * (1.0 - v) / v)
            // ((1 - v) / v).squareRoot()
            // (1 / v - 1).squareRoot()
            // (1.0 + shift * shift * (1.0 - v) / v - 1).squareRoot()
            // (shift * shift * sigma).squareRoot()
            sigma = shift * sigma
          }
          alphasCumprod.append(1 / (sigma * sigma + 1))
        }
      }
      alphasCumprod.append(1.0)
      return alphasCumprod
    }
    public func sigma(from alphaCumprod: Double) -> Double {
      if rf {
        return 1 - alphaCumprod
      } else {
        return ((1 - alphaCumprod) / alphaCumprod).squareRoot()
      }
    }
    public func alphaCumprod(from sigma: Double) -> Double {
      if rf {
        return 1 - sigma
      } else {
        return 1.0 / (sigma * sigma + 1)
      }
    }
  }
}

extension Denoiser {
  public struct LinearDiscretization: Discretization {
    public enum TimestepSpacing {
      case linspace
      case leading
      case trailing
    }
    private let internalObjective: Objective
    private let internalTimesteps: Int
    private let sigmas: [Double]
    private let timestepSpacing: TimestepSpacing
    private let EDMDiscretization: KarrasDiscretization?
    private let RFParameterization: Parameterization.RF?
    public let alphasCumprod: [Double]
    public var objective: Objective {
      if let EDMDiscretization = EDMDiscretization {
        return EDMDiscretization.objective
      }
      return internalObjective
    }
    public var timesteps: Float {
      if let EDMDiscretization = EDMDiscretization {
        return EDMDiscretization.timesteps
      }
      return Float(internalTimesteps)
    }
    init(objective: Objective, timestepSpacing: TimestepSpacing, ddpm: Parameterization.DDPM) {
      self.internalObjective = objective
      self.timestepSpacing = timestepSpacing
      internalTimesteps = ddpm.timesteps
      alphasCumprod = ddpm.alphasCumprod
      sigmas = alphasCumprod.map { ((1 - $0) / $0).squareRoot() }
      EDMDiscretization = nil
      RFParameterization = nil
    }
    init(EDMDiscretization: KarrasDiscretization) {
      self.EDMDiscretization = EDMDiscretization
      internalTimesteps = 0
      alphasCumprod = []
      sigmas = []
      internalObjective = .edm(sigmaData: 0)
      timestepSpacing = .linspace
      RFParameterization = nil
    }
    init(objective: Objective, timestepSpacing: TimestepSpacing, rf: Denoiser.Parameterization.RF) {
      self.internalObjective = objective
      self.timestepSpacing = timestepSpacing
      internalTimesteps = Int(rf.conditionScale.rounded())
      sigmas = []
      alphasCumprod = []
      EDMDiscretization = nil
      RFParameterization = rf
    }
    public init(
      _ parameterization: Parameterization, objective: Objective,
      timestepSpacing: TimestepSpacing = .linspace
    ) {
      switch parameterization {
      case .edm(_):
        self.init(EDMDiscretization: KarrasDiscretization(parameterization, objective: objective))
      case .ddpm(let ddpm):
        self.init(objective: objective, timestepSpacing: timestepSpacing, ddpm: ddpm)
      case .rf(let rf):
        self.init(objective: objective, timestepSpacing: timestepSpacing, rf: rf)
      }
    }
    public func timestep(for alphaCumprod: Double) -> Float {
      if let EDMDiscretization = EDMDiscretization {
        return EDMDiscretization.timestep(for: alphaCumprod)
      } else if let RFParameterization = RFParameterization {
        // Don't distinguish alphaCumprod and sigma any more.
        return Float((1 - alphaCumprod) * RFParameterization.conditionScale)
      }
      return timestep(for: alphaCumprod, in: sigmas)
    }
    public func alphaCumprod(timestep: Float, shift: Double) -> Double {
      if let EDMDiscretization = EDMDiscretization {
        return EDMDiscretization.alphaCumprod(timestep: timestep, shift: shift)
      } else if let RFParameterization = RFParameterization {
        var sigma = Double(timestep) / RFParameterization.conditionScale
        if shift != 1 {
          sigma = shift * sigma / (1 + (shift - 1) * sigma)
        }
        return 1 - sigma
      }
      let alphaCumprod = alphasCumprod[
        max(min(Int((timestep).rounded()), alphasCumprod.count - 1), 0)]
      if shift != 1 {
        var sigma = ((1.0 - alphaCumprod) / alphaCumprod).squareRoot()
        sigma = shift * sigma
        return 1.0 / (sigma * sigma + 1)
      }
      return alphaCumprod
    }
    public func alphasCumprod(steps: Int, shift: Double) -> [Double] {
      if let EDMDiscretization = EDMDiscretization {
        return EDMDiscretization.alphasCumprod(steps: steps, shift: shift)
      } else if let RFParameterization = RFParameterization {
        var fixedStepAlphasCumprod = [Double]()
        for i in 0..<steps {
          let timestep: Double
          switch timestepSpacing {
          case .linspace:
            // This schedule is the same for Euler A.
            timestep =
              (RFParameterization.shiftTerminal != nil
                ? Double(steps - i) / Double(steps)
                : Double(steps - 1 - i) / Double(steps - 1))
              * RFParameterization.conditionScale
          case .leading:
            // This is for DDIM the same as original stable diffusion paper.
            timestep = Double(steps - 1 - i) / Double(steps) * RFParameterization.conditionScale + 1
          case .trailing:
            // This is otherwise called "SGM Uniform".
            timestep = Double(steps - i) / Double(steps) * RFParameterization.conditionScale
          }
          var sigma = timestep / RFParameterization.conditionScale
          if shift != 1 {
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
          }
          fixedStepAlphasCumprod.append(1.0 - sigma)
        }
        if let terminal = RFParameterization.shiftTerminal, let last = fixedStepAlphasCumprod.last,
          last > 0
        {
          let scale = last / (1 - terminal)
          fixedStepAlphasCumprod = fixedStepAlphasCumprod.map { $0 / scale }
        }
        fixedStepAlphasCumprod.append(1.0)
        return fixedStepAlphasCumprod
      }
      var fixedStepAlphasCumprod = [Double]()
      for i in 0..<steps {
        let timestep: Double
        switch timestepSpacing {
        case .linspace:
          // This schedule is the same for Euler A.
          timestep = Double(steps - 1 - i) / Double(steps - 1) * Double(internalTimesteps - 1)
        case .leading:
          // This is for DDIM the same as original stable diffusion paper.
          timestep = Double(steps - 1 - i) / Double(steps) * Double(internalTimesteps) + 1
        case .trailing:
          // This is otherwise called "SGM Uniform".
          timestep = Double(steps - i) / Double(steps) * Double(internalTimesteps) - 1
        }
        let lowIdx = Int(floor(timestep))
        let highIdx = min(lowIdx + 1, internalTimesteps - 1)
        let w = timestep - Double(lowIdx)
        var sigma = exp((1 - w) * log(sigmas[lowIdx]) + w * log(sigmas[highIdx]))
        if shift != 1 {
          sigma = shift * sigma
        }
        fixedStepAlphasCumprod.append(1.0 / (sigma * sigma + 1))
      }
      fixedStepAlphasCumprod.append(1.0)
      return fixedStepAlphasCumprod
    }
    public func sigma(from alphaCumprod: Double) -> Double {
      if let _ = RFParameterization {
        return 1 - alphaCumprod
      } else {
        return ((1 - alphaCumprod) / alphaCumprod).squareRoot()
      }
    }
    public func alphaCumprod(from sigma: Double) -> Double {
      if let _ = RFParameterization {
        return 1 - sigma
      } else {
        return 1.0 / (sigma * sigma + 1)
      }
    }
  }
}

extension Denoiser {
  public struct LinearManualDiscretization: Discretization {
    private let linearDiscretization: LinearDiscretization
    private let manual: (Int) -> [Int]
    public var timesteps: Float { linearDiscretization.timesteps }
    public var objective: Objective { linearDiscretization.objective }
    public init(
      _ parameterization: Parameterization, objective: Objective,
      timestepSpacing: LinearDiscretization.TimestepSpacing = .linspace,
      manual: @escaping (Int) -> [Int]
    ) {
      linearDiscretization = LinearDiscretization(
        parameterization, objective: objective, timestepSpacing: timestepSpacing)
      self.manual = manual
    }
    public func timestep(for alphaCumprod: Double) -> Float {
      return linearDiscretization.timestep(for: alphaCumprod)
    }
    public func alphaCumprod(timestep: Float, shift: Double) -> Double {
      return linearDiscretization.alphaCumprod(timestep: timestep, shift: shift)
    }
    public func alphaCumprod(from sigma: Double) -> Double {
      return linearDiscretization.alphaCumprod(from: sigma)
    }
    public func sigma(from alphaCumprod: Double) -> Double {
      return linearDiscretization.sigma(from: alphaCumprod)
    }
    public func alphasCumprod(steps: Int, shift: Double) -> [Double] {
      let alphasCumprod = linearDiscretization.alphasCumprod
      guard !alphasCumprod.isEmpty else {
        return linearDiscretization.alphasCumprod(steps: steps, shift: shift)
      }
      let manualTimesteps = manual(steps)
      guard manualTimesteps.count == steps + 1, steps > 0 else {
        return linearDiscretization.alphasCumprod(steps: steps, shift: shift)
      }
      var fixedStepAlphasCumprod = [Double]()
      for i in 0..<steps + 1 {
        var alphaCumprod = alphasCumprod[manualTimesteps[i]]
        if shift != 1 {
          var sigma = ((1 - alphaCumprod) / alphaCumprod).squareRoot()
          sigma = shift * sigma
          alphaCumprod = 1.0 / (sigma * sigma + 1)
        }
        fixedStepAlphasCumprod.append(alphaCumprod)
      }
      fixedStepAlphasCumprod.append(1.0)
      return fixedStepAlphasCumprod
    }
  }
}

extension Denoiser {
  public struct AYSLogLinearInterpolatedKarrasDiscretization: Discretization {
    private let karrasDiscretization: KarrasDiscretization
    private let samplingSigmas: [Double]
    public var timesteps: Float { karrasDiscretization.timesteps }
    public var objective: Objective { karrasDiscretization.objective }
    public init(
      _ parameterization: Parameterization, objective: Objective,
      samplingSigmas: [Double]
    ) {
      karrasDiscretization = KarrasDiscretization(
        parameterization, objective: objective)
      self.samplingSigmas = samplingSigmas
    }
    public func timestep(for alphaCumprod: Double) -> Float {
      return karrasDiscretization.timestep(for: alphaCumprod)
    }
    public func alphaCumprod(timestep: Float, shift: Double) -> Double {
      return karrasDiscretization.alphaCumprod(timestep: timestep, shift: shift)
    }
    public func alphaCumprod(from sigma: Double) -> Double {
      return karrasDiscretization.alphaCumprod(from: sigma)
    }
    public func sigma(from alphaCumprod: Double) -> Double {
      return karrasDiscretization.sigma(from: alphaCumprod)
    }
    private static func logLinearInterpolation(sigmas: [Double], steps: Int) -> [Double] {
      guard steps + 1 != sigmas.count else {
        return sigmas
      }
      var ys = [Double]()
      for i in 0..<(sigmas.count - 1) {
        ys.append(log(Double(sigmas[sigmas.count - 2 - i])))
      }
      var scaledReverseSigmas = [Double]()
      for i in 0..<steps {
        let y = Double(i) / Double(steps - 1) * Double(ys.count - 1)
        let y0 = max(Int(y.rounded(.down)), 0)
        let y1 = min(Int(y.rounded(.up)), ys.count - 1)
        let a = y - Double(y0)
        let scaledYs = (1 - a) * ys[y0] + a * ys[y1]
        scaledReverseSigmas.append(exp(scaledYs))
      }
      return scaledReverseSigmas.reversed() + [0]
    }
    public func alphasCumprod(steps: Int, shift: Double) -> [Double] {
      guard !samplingSigmas.isEmpty else {
        return karrasDiscretization.alphasCumprod(steps: steps, shift: shift)
      }
      let scaledSigmas = Self.logLinearInterpolation(sigmas: samplingSigmas, steps: steps)

      guard scaledSigmas.count == steps + 1, steps > 0 else {
        return karrasDiscretization.alphasCumprod(steps: steps, shift: shift)
      }
      var alphasCumprod = [Double]()
      for i in 0..<steps {
        var sigma = scaledSigmas[i]
        if shift != 1 {
          // Simplify Sigmoid[Log[x / (1 - x)] + 2 * Log[1 / shift]]
          // var v: Double = 1.0 / (sigma * sigma + 1)
          // v = 1.0 / (1.0 + shift * shift * (1.0 - v) / v)
          // ((1 - v) / v).squareRoot()
          // (1 / v - 1).squareRoot()
          // (1.0 + shift * shift * (1.0 - v) / v - 1).squareRoot()
          // (shift * shift * sigma).squareRoot()
          sigma = shift * sigma
        }
        alphasCumprod.append(1 / (sigma * sigma + 1))
      }
      alphasCumprod.append(1.0)
      return alphasCumprod
    }
  }
}

extension Denoiser {
  public struct AYSLogLinearInterpolatedTimestepDiscretization: Discretization {
    private let linearDiscretization: LinearDiscretization
    private let samplingTimesteps: [Int]
    public var timesteps: Float { linearDiscretization.timesteps }
    public var objective: Objective { linearDiscretization.objective }
    public init(
      _ parameterization: Parameterization, objective: Objective,
      timestepSpacing: LinearDiscretization.TimestepSpacing = .linspace,
      samplingTimesteps: [Int]
    ) {
      linearDiscretization = LinearDiscretization(
        parameterization, objective: objective, timestepSpacing: timestepSpacing)
      self.samplingTimesteps = samplingTimesteps
    }
    public func timestep(for alphaCumprod: Double) -> Float {
      return linearDiscretization.timestep(for: alphaCumprod)
    }
    public func alphaCumprod(timestep: Float, shift: Double) -> Double {
      return linearDiscretization.alphaCumprod(timestep: timestep, shift: shift)
    }
    public func alphaCumprod(from sigma: Double) -> Double {
      return linearDiscretization.alphaCumprod(from: sigma)
    }
    public func sigma(from alphaCumprod: Double) -> Double {
      return linearDiscretization.sigma(from: alphaCumprod)
    }
    private func logLinearInterpolation(timesteps: [Int], steps: Int) -> [Int] {
      guard steps + 1 != timesteps.count else {
        return timesteps
      }
      var ys = [Double]()
      for i in 0..<(timesteps.count - 1) {
        let alphaCumprod = linearDiscretization.alphaCumprod(
          timestep: Float(timesteps[timesteps.count - 2 - i]), shift: 1)
        let sigma = ((1.0 - alphaCumprod) / alphaCumprod).squareRoot()
        ys.append(log(sigma))
      }
      var scaledReverseTimesteps = [Int]()
      for i in 0..<steps {
        let y = Double(i) / Double(max(1, steps - 1)) * Double(ys.count - 1)
        let y0 = max(Int(y.rounded(.down)), 0)
        let y1 = min(Int(y.rounded(.up)), ys.count - 1)
        let a = y - Double(y0)
        let scaledYs = (1 - a) * ys[y0] + a * ys[y1]
        let sigma = exp(scaledYs)
        let alphaCumprod = 1.0 / (sigma * sigma + 1)
        let timestep = linearDiscretization.timestep(for: alphaCumprod)
        scaledReverseTimesteps.append(Int(timestep.rounded()))
      }
      return scaledReverseTimesteps.reversed() + [0]
    }
    public func alphasCumprod(steps: Int, shift: Double) -> [Double] {
      let alphasCumprod = linearDiscretization.alphasCumprod
      guard !alphasCumprod.isEmpty && !samplingTimesteps.isEmpty else {
        return linearDiscretization.alphasCumprod(steps: steps, shift: shift)
      }
      let scaledTimesteps = logLinearInterpolation(timesteps: samplingTimesteps, steps: steps)
      guard scaledTimesteps.count == steps + 1, steps > 0 else {
        return linearDiscretization.alphasCumprod(steps: steps, shift: shift)
      }
      var fixedStepAlphasCumprod = [Double]()
      for i in 0..<steps + 1 {
        var alphaCumprod = alphasCumprod[scaledTimesteps[i]]
        if shift != 1 {
          var sigma = ((1 - alphaCumprod) / alphaCumprod).squareRoot()
          sigma = shift * sigma
          alphaCumprod = 1.0 / (sigma * sigma + 1)
        }
        fixedStepAlphasCumprod.append(alphaCumprod)
      }
      fixedStepAlphasCumprod.append(1.0)
      return fixedStepAlphasCumprod
    }
  }
}

extension Denoiser.Discretization {
  public func sigmas(steps: Int, shift: Double = 1.0) -> [Double] {
    return alphasCumprod(steps: steps, shift: shift).map { ((1 - $0) / $0).squareRoot() }
  }

  fileprivate func timestep(for alphaCumprod: Double, in sigmas: [Double]) -> Float {
    let sigma = ((1 - alphaCumprod) / alphaCumprod).squareRoot()
    guard sigma > sigmas[0] else {
      return 0
    }
    guard sigma < sigmas[sigmas.count - 1] else {
      return Float(sigmas.count - 1)
    }
    // Find in between which sigma resides.
    var highIdx = sigmas.count - 1
    var lowIdx = 0
    while lowIdx < highIdx - 1 {
      let midIdx = lowIdx + (highIdx - lowIdx) / 2
      if sigma < sigmas[midIdx] {
        highIdx = midIdx
      } else {
        lowIdx = midIdx
      }
    }
    assert(sigma >= sigmas[highIdx - 1] && sigma <= sigmas[highIdx])
    let low = log(sigmas[highIdx - 1])
    let high = log(sigmas[highIdx])
    let logSigma = log(sigma)
    let w = min(max((low - logSigma) / (low - high), 0), 1)
    return Float((1.0 - w) * Double(highIdx - 1) + w * Double(highIdx))
  }

  fileprivate func alphaCumprod(timestep: Float, in sigmas: [Double]) -> Double {
    guard timestep > 0 else {
      return 1.0 / (sigmas[0] * sigmas[0] + 1)
    }
    guard timestep < Float(sigmas.count - 1) else {
      return 1.0 / (sigmas[sigmas.count - 1] * sigmas[sigmas.count - 1] + 1)
    }
    // Find in between which sigma resides.
    let low = log(sigmas[min(sigmas.count - 1, max(0, Int(timestep.rounded(.down))))])
    let high = log(sigmas[min(sigmas.count - 1, max(0, Int(timestep.rounded(.up))))])
    let w = Double(min(max(timestep - timestep.rounded(.down), 0), 1))
    let logSigma = low * (1 - w) + w * high
    let sigma = exp(logSigma)
    return 1.0 / (sigma * sigma + 1)
  }
}
