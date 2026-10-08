import Crypto
import Foundation
import Logging

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

public actor ProxyMessageSigner {
  // In-memory key pair
  private var keyPair: (private: P256.Signing.PrivateKey, public: P256.Signing.PublicKey)

  init() {
    // Generate initial key pair in memory
    let privateKey = P256.Signing.PrivateKey()
    let publicKey = privateKey.publicKey
    self.keyPair = (private: privateKey, public: publicKey)
  }

  // MARK: - Public Interface

  /// Sign a message with private key
  public func signMessage(_ message: String) -> Data? {
    let messageData = Data(message.utf8)
    guard let signature = try? keyPair.private.signature(for: messageData) else {
      return nil
    }
    return signature.rawRepresentation
  }

  /// Verify signature with public key
  public func verifySignature(_ signatureData: Data, for message: String) -> Bool {
    let messageData = Data(message.utf8)
    guard let signature = try? P256.Signing.ECDSASignature(rawRepresentation: signatureData) else {
      return false
    }
    return keyPair.public.isValidSignature(signature, for: messageData)
  }

  private func createDirectJWT(_ payload: Data) throws -> String {
    // Create header
    let header = ["alg": "ES256", "typ": "JWT"]
    let headerData = try JSONSerialization.data(withJSONObject: header)
    let headerBase64 = base64URLEncode(headerData)
    let payloadBase64 = base64URLEncode(payload)

    let headerAndPayload = "\(headerBase64).\(payloadBase64)"
    let messageData = Data(headerAndPayload.utf8)
    let signature = try keyPair.private.signature(for: SHA256.hash(data: messageData))
    let signatureBase64 = base64URLEncode(signature.rawRepresentation)

    return "\(headerAndPayload).\(signatureBase64)"
  }

  /// Verify and decode a direct JWT token (returns the payload dictionary)
  public func verifyDirectJWT(_ token: String) -> [String: Any]? {
    let parts = token.components(separatedBy: ".")
    guard parts.count == 3 else { return nil }

    let headerAndPayload = "\(parts[0]).\(parts[1])"

    // Decode signature
    guard let signatureData = base64URLDecode(parts[2]),
      let signature = try? P256.Signing.ECDSASignature(rawRepresentation: signatureData)
    else {
      return nil
    }

    // Verify signature
    let messageData = Data(headerAndPayload.utf8)
    guard keyPair.public.isValidSignature(signature, for: SHA256.hash(data: messageData)) else {
      return nil
    }

    // Decode payload to extract the direct payload
    guard let payloadData = base64URLDecode(parts[1]),
      let payloadDict = try? JSONSerialization.jsonObject(with: payloadData) as? [String: Any]
    else {
      return nil
    }

    return payloadDict
  }

  /// Get public key as PEM string
  public func getPublicKey() -> String? {
    return keyPair.public.pemRepresentation
  }

  /// Generate new key pair in memory
  public func reloadKeys() {
    let privateKey = P256.Signing.PrivateKey()
    let publicKey = privateKey.publicKey
    self.keyPair = (private: privateKey, public: publicKey)
  }

  // MARK: - Helper Methods

  private func base64URLEncode(_ data: Data) -> String {
    let base64 = data.base64EncodedString()
    return
      base64
      .replacingOccurrences(of: "+", with: "-")
      .replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "=", with: "")
  }

  private func base64URLDecode(_ string: String) -> Data? {
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

  public enum CompletionCode: String, Codable {
    case cancel
    case complete
  }

  public func completeBoost(
    action: CompletionCode, generationId: String, amount: Int, logger: Logger
  ) async {
    do {
      struct Request: Codable {
        var action: CompletionCode
        var generationId: String
        var amount: Int
      }
      let request = Request(action: action, generationId: generationId, amount: amount)
      let jsonEncoder = JSONEncoder()
      jsonEncoder.outputFormatting = [.sortedKeys]
      let payload = try jsonEncoder.encode(request)
      let jwtToken = try createDirectJWT(payload)

      await completeConsumableGeneration(
        token: jwtToken, generationId: generationId, logger: logger)
    } catch {
      logger.error("Failed to process completion: \(error)")
    }
  }

  public func completePayg(
    action: CompletionCode, generationId: String, amount: Int, logger: Logger
  ) async {
    do {
      struct Request: Codable {
        var action: CompletionCode
        var generationId: String
        var amount: Int
      }
      let request = Request(action: action, generationId: generationId, amount: amount)
      let jsonEncoder = JSONEncoder()
      jsonEncoder.outputFormatting = [.sortedKeys]
      let payload = try jsonEncoder.encode(request)
      let jwtToken = try createDirectJWT(payload)

      await completeConsumableGeneration(
        token: jwtToken, generationId: generationId, logger: logger)
    } catch {
      logger.error("Failed to process completion: \(error)")
    }
  }

  private func completeConsumableGeneration(
    token: String, generationId: String, logger: Logger
  ) async {
    do {
      // Create the URL
      guard let url = URL(string: "https://api.drawthings.ai/complete_consumable_generation") else {
        logger.error("Invalid URL for complete_consumable_generation")
        return
      }

      // Create the request
      var request = URLRequest(url: url)
      request.httpMethod = "POST"
      request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")

      logger.info("Sending request to \(url.absoluteString)")
      logger.info("Authorization header: Bearer \(token)")

      let (_, response) = try await URLSession.shared.data(for: request)

      guard let httpResponse = response as? HTTPURLResponse else {
        logger.info("Invalid response: \(String(describing: type(of: response)))")
        return
      }
      logger.info("Response status code: \(httpResponse.statusCode)")
      logger.info("Response status: \(httpResponse)")

      if httpResponse.statusCode == 200 {
        logger.info(
          "Complete consumable generation request succeeded, generationId: \(generationId)")
      } else {
        logger.error(
          "Complete consumable generation request failed with status code: \(httpResponse.statusCode), generationId: \(generationId)"
        )
      }
    } catch {
      logger.error("Failed to send request to complete_consumable_generation: \(error)")
    }
  }
}
