import Foundation
import CoreBluetooth

/// CoreBluetooth implementation of `MimoTransport`.
///
/// NOT IMPLEMENTED YET. Skeleton exists so the pipeline can be written against
/// a real type.
///
/// TODO(ble):
///   1. Add to Info.plist (host app, not this SPM target):
///        NSBluetoothAlwaysUsageDescription = "Connect to your Mimo bot"
///   2. `CBCentralManager` with a delegate; start scan with
///        scanForPeripherals(withServices: [MimoBotIds.service], options: nil)
///   3. On `didConnect`, request MTU (iOS handles automatically), discover
///        service + characteristics, subscribe to notifications on
///        AUDIO_UP + CONTROL.
///   4. Strip the 2-byte LE seq prefix from AUDIO_UP payloads before yielding
///        onto `audioUp`; prepend on `sendAudioDown`.
///   5. Use `.writeWithoutResponse` for AUDIO_DOWN; track
///        `peripheral.canSendWriteWithoutResponse` + the
///        `peripheralIsReadyToSendWriteWithoutResponse` callback so the stream
///        doesn't outrun BLE throughput.
final class BLETransport: NSObject, MimoTransport {
    var state: AsyncStream<MimoTransportState> { stateStream }
    var audioUp: AsyncStream<Data> { audioUpStream }
    var control: AsyncStream<String> { controlStream }

    private let (stateStream, stateCont)     = AsyncStream.makeStream(of: MimoTransportState.self)
    private let (audioUpStream, audioUpCont) = AsyncStream.makeStream(of: Data.self)
    private let (controlStream, controlCont) = AsyncStream.makeStream(of: String.self)

    private var central: CBCentralManager?
    private var peripheral: CBPeripheral?
    private var audioDownChar: CBCharacteristic?
    private var controlChar: CBCharacteristic?
    private var outgoingSeq: UInt16 = 0

    override init() {
        super.init()
        // CBCentralManager init happens on connect(); queueing it early also works.
    }

    func connect() async throws {
        stateCont.yield(.scanning)
        fatalError("TODO(ble): start scan for MimoBotIds.service, connect, discover, subscribe")
    }

    func sendAudioDown(_ opusFrame: Data) async throws {
        fatalError("TODO(ble): prepend 2-byte LE seq, write-without-response to audioDownChar")
    }

    func sendControl(_ json: String) async throws {
        fatalError("TODO(ble): write json.data(using: .utf8) to controlChar")
    }

    func disconnect() async {
        if let p = peripheral { central?.cancelPeripheralConnection(p) }
        peripheral = nil
        stateCont.yield(.disconnected)
    }
}
