import Crypto
import Foundation

public class JWTDecoder {
  private var publicKeyPEM: String
  public init(publicKeyPEM: String) throws {
    self.publicKeyPEM = publicKeyPEM
  }

  public func decode(_ token: String) throws -> JWTPayload {
    // Split JWT parts
    let parts = token.components(separatedBy: ".")
    guard parts.count == 3 else {
      throw JWTError.invalidToken
    }

    let payload = parts[1]

    // Verify signature
    guard try verifyES256JWT(token: token, publicKeyPEM: publicKeyPEM) else {
      throw JWTError.signatureVerificationFailed("failed signature")
    }
    // Decode payload
    return try decodePayload(payload)
  }

  func verifyES256JWT(token: String, publicKeyPEM: String) throws -> Bool {
    // Split the JWT into parts
    let parts = token.components(separatedBy: ".")
    guard parts.count == 3 else {
      throw JWTError.invalidToken
    }

    let headerAndPayload = parts[0] + "." + parts[1]
    let signature = parts[2]

    // Decode the signature from base64url to Data
    guard let signatureData = base64URLDecode(signature) else {
      throw JWTError.invalidBase64
    }

    // Convert the message to Data
    guard let messageData = headerAndPayload.data(using: .utf8) else {
      throw JWTError.invalidToken
    }

    // Create P256 key from DER/ASN.1 format
    guard let p256PublicKey = try? P256.Signing.PublicKey(pemRepresentation: publicKeyPEM) else {
      throw JWTError.invalidPublicKey
    }

    // Create P256.Signing.ECDSASignature from raw signature
    guard let ecdsaSignature = try? P256.Signing.ECDSASignature(rawRepresentation: signatureData)
    else {
      throw JWTError.invalidSignature
    }

    // Verify the signature
    return p256PublicKey.isValidSignature(ecdsaSignature, for: SHA256.hash(data: messageData))
  }

  // Helper function to decode base64url to Data
  func base64URLDecode(_ string: String) -> Data? {
    var base64 =
      string
      .replacingOccurrences(of: "-", with: "+")
      .replacingOccurrences(of: "_", with: "/")

    // Add padding if needed
    while base64.count % 4 != 0 {
      base64 += "="
    }

    return Data(base64Encoded: base64)
  }

  private func decodePayload(_ payload: String) throws -> JWTPayload {
    guard let payloadData = base64URLDecode(payload) else {
      throw JWTError.invalidPayload
    }

    return try JSONDecoder().decode(JWTPayload.self, from: payloadData)
  }
}

public enum UserClass: String, Codable {
  case plus = "plus"
  case community = "community"
  case background = "background"
  case banned = "banned"
  case throttled = "throttled"
}

public enum ConsumableType: String, Codable {
  case boost = "boost"
  case payg = "payg"
  case paygFree = "payg_free"
}

// JWT Payload structure matching your token
public struct JWTPayload: Codable {
  public let checksum: String
  public let stats: [String: Int]
  public let nonce: String
  public let iss: String
  public let exp: Int
  // These are for the consumable.
  public let generationId: String?
  public let amount: Int?
  public let consumableType: ConsumableType?
  public let fromBridge: Bool?  // request from sdk / api
  public let userId: String?
  public let userClass: UserClass?
}

enum JWTError: Error {
  case invalidPublicKey
  case invalidToken
  case invalidSignature
  case invalidPayload
  case invalidBase64
  case signatureVerificationFailed(String)
  case opensslError(String)

  var localizedDescription: String {
    switch self {
    case .invalidPublicKey:
      return "Invalid public key format"
    case .invalidToken:
      return "Invalid token format"
    case .invalidSignature:
      return "Invalid token signature"
    case .invalidPayload:
      return "Invalid token payload"
    case .invalidBase64:
      return "Invalid base64 encoding"
    case .signatureVerificationFailed(let message):
      return "Signature verification failed: \(message)"
    case .opensslError(let message):
      return "Open SSL failed: \(message)"
    }
  }
}
