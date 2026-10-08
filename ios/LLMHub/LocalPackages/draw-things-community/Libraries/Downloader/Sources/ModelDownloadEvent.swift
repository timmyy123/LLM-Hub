import Foundation

/// Download lifecycle events for hosts that display progress alongside command output.
public enum ModelDownloadEvent {
  case started(
    id: UUID, name: String, subtitle: String, files: [String], cancel: () -> Void)
  case progress(
    id: UUID, file: String, index: Int, totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64)
  case finished(id: UUID)
}
