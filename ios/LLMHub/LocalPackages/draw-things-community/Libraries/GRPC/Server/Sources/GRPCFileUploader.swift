import Crypto
import Foundation
import GRPC
import GRPCImageServiceModels
import Logging
import NIO

public enum GRPCFileUploaderError: Error {
  case notConnected
  case cannotReadFile
}

public struct GRPCFileUploader {
  public typealias Call = BidirectionalStreamingCall<FileUploadRequest, UploadResponse>
  private static let batchSize = 50  // We send 50MiB at once (i.e. 50 chunks).
  private static let chunkSize = 1 * 1024 * 1024  // 1MiB chunks.
  private let logger = Logger(label: "com.draw-things.grpc-file-uploader")
  public let client: ImageGenerationClientWrapper

  public init(client: ImageGenerationClientWrapper) {
    self.client = client
  }

  public func resume(
    fileUrl: URL,
    progressHandler: @escaping (Int64, Int64) -> Void,
    completionHandler: @escaping (Bool) -> Void
  ) throws -> Call {
    let sharedSecret = client.sharedSecret
    guard let client = client.client else {
      throw GRPCFileUploaderError.notConnected
    }

    guard let fileData = try? Data(contentsOf: fileUrl, options: .mappedIfSafe) else {
      throw GRPCFileUploaderError.cannotReadFile
    }

    let filename = fileUrl.lastPathComponent
    let fileSize = Int64(fileData.count)

    let logger = logger
    var call: BidirectionalStreamingCall<FileUploadRequest, UploadResponse>? = nil
    var sentOffset: Int64 = 0
    var offset: Int64 = 0
    // Create the stream for sending data to the server
    let callInstance = client.uploadFile { response in
      // 2. Break the file into chunks and send them sequentially.
      logger.info("server received chunk offset \(response.receivedOffset)")
      progressHandler(response.receivedOffset, fileSize)

      guard response.chunkUploadSuccess else {
        logger.info("failed to upload chunk \(response.message)")
        call?.cancel(promise: nil)
        return
      }
      guard response.receivedOffset == sentOffset else {
        // This is when we received responses that less than what we sent.
        return
      }
      // Create a chunk of data
      for _ in 0..<Self.batchSize {
        // Break if offset is bigger.
        if offset >= fileSize {
          break
        }
        let chunkData = fileData.subdata(
          in: Int(offset)..<min(Int(offset + Int64(Self.chunkSize)), fileData.count))

        // Create a FileChunk message
        let chunkRequest = FileUploadRequest.with {
          $0.chunk = FileChunk.with {
            $0.content = chunkData
            $0.offset = offset
          }
          if let sharedSecret = sharedSecret {
            $0.sharedSecret = sharedSecret
          }
        }
        // Send the chunk to the server
        sentOffset = offset
        let _ = call?.sendMessage(chunkRequest)
        logger.info("Sent chunk at offset \(offset)")
        // Increment the offset
        offset = Int64(min(Int(offset + Int64(Self.chunkSize)), fileData.count))
      }
      guard offset >= fileSize else { return }
      // Send the last call.
      let _ = call?.sendEnd()
    }
    call = callInstance

    callInstance.status.whenComplete { result in
      switch result {
      case .success(let status):
        completionHandler(status == .ok)

      case .failure(let error):
        logger.error("Failed to send initRequest: \(error)")
        completionHandler(false)
      }
    }

    callInstance.eventLoop.execute {
      // Send message and compute SHA-256 off the main thread to the event loop queue.
      let fileHash = Data(SHA256.hash(data: fileData))
      let hexString = fileHash.map { String(format: "%02x", $0) }.joined()
      logger.info("file hash \(hexString)")
      // 1. Send the InitUploadRequest with filename, hash, and total file size
      let initRequest = FileUploadRequest.with {
        $0.initRequest = InitUploadRequest.with {
          $0.filename = filename
          $0.sha256 = fileHash
          $0.totalSize = fileSize
        }
        if let sharedSecret = sharedSecret {
          $0.sharedSecret = sharedSecret
        }
      }
      callInstance.sendMessage(initRequest).whenComplete { result in
        switch result {
        case .success:
          logger.info("Successfully sent init request")
        case .failure(let error):
          callInstance.cancel(promise: nil)
          logger.error("Failed to send initRequest: \(error)")
          completionHandler(false)
        }
      }
    }
    return callInstance
  }

}
