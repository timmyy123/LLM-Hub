import Foundation

// MARK: - Types moved from AccountManager+DeviceAttestation.swift

public struct ChallengeResponse: Codable {
  public let challenge: String

  public init(challenge: String) {
    self.challenge = challenge
  }
}

public struct AttestationRequest: Codable {
  public let attestation: String
  public let keyId: String
  public let challenge: String

  public init(attestation: String, keyId: String, challenge: String) {
    self.attestation = attestation
    self.keyId = keyId
    self.challenge = challenge
  }
}

public struct AssertionPayload: Codable {
  public let assertion: String
  public let keyId: String

  public init(assertion: String, keyId: String) {
    self.assertion = assertion
    self.keyId = keyId
  }
}

public struct VerificationResponse: Codable {
  public let success: Bool
  public let message: String

  public init(success: Bool, message: String) {
    self.success = success
    self.message = message
  }
}
