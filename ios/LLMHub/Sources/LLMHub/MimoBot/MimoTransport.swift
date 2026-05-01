import Foundation

/// Abstraction over the link to the Mimo bot. v0 is BLE.
///
/// Frames on `audioUp` / `sendAudioDown` are Opus payloads. Sequence numbers
/// and GATT framing live one level below this protocol.
protocol MimoTransport: AnyObject {
    var state: AsyncStream<MimoTransportState> { get }
    var audioUp: AsyncStream<Data> { get }
    var control: AsyncStream<String> { get }

    func connect() async throws
    func sendAudioDown(_ opusFrame: Data) async throws
    func sendControl(_ json: String) async throws
    func disconnect() async
}

enum MimoTransportState: Sendable {
    case disconnected
    case scanning
    case connecting
    case connected
    case error(String)
}
